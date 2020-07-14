# ----------------------------------------------------------------------------
#
# Phiflow Karman vortex solver framework
# Copyright 2020 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Data generation
#
# ----------------------------------------------------------------------------

import os, sys, logging, argparse, pickle, glob, random, distutils.dir_util

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

from PIL import Image           # for writing PNGs
def save_img(array, scale, name, idx=0):
    if len(array.shape) <= 4:
        ima = np.reshape(array[idx], [array.shape[1], array.shape[2]])  # remove channel dimension, 2d
    else:
        ima = array[idx, :, array.shape[1] // 2, :, 0]  # 3d, middle z slice

    ima = np.reshape(ima, [array.shape[1], array.shape[2]])  # remove channel dimension
    # ima = ima[::-1, :]  # flip along y
    image = Image.fromarray(np.asarray(ima * scale, dtype='i'))
    print("\tWriting image: " + name)
    image.save(name)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',             default='0',             help='visible GPUs')
parser.add_argument('--cuda',            action='store_true',     help='enable CUDA for solver')
parser.add_argument('-o', '--output',    default=None,            help='path to an output directory')
parser.add_argument('--thumb',           action='store_true',     help='save thumbnail images')
parser.add_argument('-t', '--simsteps',  default=1500, type=int,  help='simulation steps: an epoch')
parser.add_argument('-s', '--skipsteps', default=999, type=int,   help='skip first steps; (vortices may not form)')
parser.add_argument('-r', '--res',       default=32, type=int,    help='resolution of the reference axis')
parser.add_argument('--re',              default=1e6, type=float, help='Effective Reynolds number')
parser.add_argument('--initdH',          default=None,            help='load hires (will be downsampled) density  (e.g., dens_0000.npz)')
parser.add_argument('--initvH',          default=None,            help='load hires (will be downsampled) velocity (e.g., velo_0000.npz)')
parser.add_argument('-d', '--scale',     default=4, type=int,     help='down-sampling scale of hires (only valid when initdH or initvH given)')
parser.add_argument('-l', '--len',       default=100, type=int,   help='length of the reference axis')
parser.add_argument('--seed',            default=0, type=int,     help='seed for random number generator')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

if params['cuda']: from phi.tf.tf_cuda_pressuresolver import CUDASolver

from phi.tf.flow import *
import phi.tf.util

import tensorflow as tf
from tensorflow import keras

random.seed(params['seed'])
np.random.seed(params['seed'])
tf.compat.v1.set_random_seed(params['seed'])

def downsample4x(tensor):
    return math.downsample2x(math.downsample2x(tensor))

def downsample4xSMAC(tensor):
    return StaggeredGrid(tensor).downsample2x().downsample2x().staggered_tensor()

class KarmanFlow(IncompressibleFlow):
    def __init__(self, pressure_solver=None, make_input_divfree=False, make_output_divfree=True):
        IncompressibleFlow.__init__(self, pressure_solver, make_input_divfree, make_output_divfree)

        self.infl = Inflow(box[5:10, 25:75])
        self.obst = Obstacle(Sphere([50, 50], 10))

    def step(self, smoke, re, res, velBCy, velBCyMask, dt=1.0, gravity=Gravity()):
        # apply viscosity
        alpha = 1.0/re * dt * res * res

        vel = smoke.velocity.data
        cy = diffuse(CenteredGrid(vel[0].data), alpha)
        cx = diffuse(CenteredGrid(vel[1].data), alpha)

        # apply velocity BCs, only y for now; velBCy should be pre-multiplied
        cy = cy*(1.0 - velBCyMask) + velBCy

        smoke = smoke.copied_with(velocity=StaggeredGrid([cy.data, cx.data], smoke.velocity.box))

        return super().step(fluid=smoke, dt=dt, obstacles=[self.obst], gravity=gravity, density_effects=[self.infl], velocity_effects=())


downDen = eval('downsample{}x'.format(params['scale']))
downVel = eval('downsample{}xSMAC'.format(params['scale']))

st = Fluid(Domain(resolution=[params['res']*2, params['res']], box=box[0:params['len']*2, 0:params['len']], boundaries=OPEN), buoyancy_factor=0)

# init density & velocity

d0 = downDen(read_zipped_array(params['initdH'])) if params['initdH'] else CenteredGrid(st.density.data, st.density.box)
if params['initvH']:
    vn = downVel(read_zipped_array(params['initvH']))  # NOTE: read_zipped_array reverts indices! [...,0]=u, [...,1]=v -> [...,0]=v, [...,1]=u
    v0 = StaggeredGrid(unstack_staggered_tensor(vn), st.velocity.box)
else:
    vn = st.velocity.staggered_tensor()
    vn[..., 0] = 1.0                                                                         # warm start - initialize flow to 1 along y everywhere
    vn[..., vn.shape[1]//2+10:vn.shape[1]//2+20, vn.shape[2]//2-2:vn.shape[2]//2+2, 1] = 1.0  # modify x, poke sideways to trigger instability
    v0 = StaggeredGrid(unstack_staggered_tensor(vn), st.velocity.box)

st = st.copied_with(density=d0, velocity=v0)

# velocity BC
vn = np.zeros(st.velocity.data[0].data.shape)  # NOTE: st.velocity.data[0] is considered as the velocity field in y axis!
vn[..., 0:2, 0:vn.shape[2]-1, 0] = 1.0
vn[..., 0:vn.shape[1], 0:1,   0] = 1.0
vn[..., 0:vn.shape[1], -1:,   0] = 1.0
velBCy = vn
velBCyMask = np.copy(vn)        # warning, only works for 1s, otherwise setup/scale

# phiflow scene

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tf_session = tf.Session(config=config)

scene = Scene.create(directory=params['output'])
sess  = Session(scene, session=tf_session)

log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))
log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

if params['output']:
    with open(os.path.normpath(scene.path)+'/params.pickle', 'wb') as f: pickle.dump(params, f)

simulator = KarmanFlow()
tf_st_in = phi.tf.util.placeholder_like(st)
tf_st = simulator.step(tf_st_in, re=params['re'], res=params['res'], velBCy=velBCy, velBCyMask=velBCyMask)

if params['skipsteps']==0 and params['output'] is not None:
    scene.write(
        [st.density, st.velocity],
        ['dens', 'velo'],
        0
    )

for i in range(1, params['simsteps']):
    my_feed_dict = { tf_st_in: st }
    st = sess.run(tf_st, my_feed_dict)

    log.info('Step {:06d}'.format(i))
    if params['skipsteps']<i and params['output'] is not None:
        scene.write(
            [st.density, st.velocity],
            ['dens', 'velo'],
            i
        )
        if params['thumb'] and params['output'] is not None:
            thumb_path = os.path.normpath(scene.path).replace(os.path.basename(scene.path), "thumb/{}".format(os.path.basename(scene.path)))
            distutils.dir_util.mkpath(thumb_path)
            save_img(st.density.data,          10000., thumb_path + "/dens_{:06d}.png".format(i))
            save_img(st.velocity.data[1].data, 10000., thumb_path + "/velU_{:06d}.png".format(i))
            save_img(st.velocity.data[0].data, 10000., thumb_path + "/velV_{:06d}.png".format(i))
