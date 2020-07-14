# ----------------------------------------------------------------------------
#
# Phiflow Karman vortex solver framework
# Copyright 2020 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Apply correction network model
#
# ----------------------------------------------------------------------------

import os, sys, logging, argparse, pickle

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',            default='0',                                  help='visible GPUs')
parser.add_argument('-s', '--scale',    default=4, type=int,                          help='simulation scale for high-res')
parser.add_argument('-r', '--res',      default=32, type=int,                         help='resolution of the reference axis')
parser.add_argument('-l', '--len',      default=100, type=int,                        help='length of the reference axis')
parser.add_argument('--re',             default=1e6, type=float,                      help='Reynolds number')
parser.add_argument('--initdH',         default=None,                                 help='load hires (will be downsampled) density  (e.g., dens_0000.npz)')
parser.add_argument('--initvH',         default=None,                                 help='load hires (will be downsampled) velocity (e.g., velo_0000.npz)')
parser.add_argument('-t', '--simsteps', default=500, type=int,                        help='simulation steps')
parser.add_argument('-o', '--output',   default='/tmp/phiflow/run',                   help='path to an output directory')
parser.add_argument('--stats',          default='/tmp/phiflow/data/dataStats.pickle', help='path to datastats')
parser.add_argument('--model',          default='/tmp/phiflow/tf/model.h5',           help='path to a tensorflow model')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

from phi.flow import *

import tensorflow as tf
from tensorflow import keras

def to_feature(smokestates, ext_const_channel):
    # input feature; drop the unused edges of the staggered velocity grid making its dim same to the centered grid's
    with tf.name_scope('to_feature') as scope:
        return math.concat(
            [smokestates[j].velocity.staggered_tensor()[:, :-1:, :-1:, 0:2] for j in range(len(smokestates))] +
            [np.ones(shape=smokestates[0].density.data.shape)*ext_const_channel],  # Reynolds number
            axis=-1
        )

def to_staggered(tensor_cen, box):
    with tf.name_scope('to_staggered') as scope:
        return StaggeredGrid(math.pad(tensor_cen, ((0,0), (0,1), (0,1), (0,0))), box=box)

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

        cx = diffuse(CenteredGrid(smoke.velocity.data[1].data), alpha)
        cy = diffuse(CenteredGrid(smoke.velocity.data[0].data), alpha)

        # apply velocity BCs, only x for now; velBCx should be pre-multiplied
        cy = cy*(1.0 - velBCyMask) + velBCy

        smoke = smoke.copied_with(velocity=StaggeredGrid([cy.data, cx.data], smoke.velocity.box))

        return super().step(fluid=smoke, dt=dt, obstacles=[self.obst], gravity=gravity, density_effects=[self.infl], velocity_effects=())


downDen = eval('downsample{}x'.format(params['scale']))
downVel = eval('downsample{}xSMAC'.format(params['scale']))

# initialize

st = Fluid(Domain(resolution=[params['res']*2, params['res']], box=box[0:params['len']*2, 0:params['len']], boundaries=OPEN), buoyancy_factor=0)

d0 = downDen(read_zipped_array(params['initdH'])) if params['initdH'] else CenteredGrid(st.density.data, st.density.box)
if params['initvH']:
    vn = downVel(read_zipped_array(params['initvH']))  # NOTE: read_zipped_array reverts indices! [...,0]=u, [...,1]=v -> [...,0]=v, [...,1]=u
    v0 = StaggeredGrid(unstack_staggered_tensor(vn), st.velocity.box)
else:
    vn = st.velocity.staggered_tensor()
    vn[..., 0] = 1.0                                                                         # warm start - initialize flow to 1 along y everywhere
    vn[..., vn.shape[1]//2+10:vn.shape[1]//2+20, vn.shape[2]//2-2:vn.shape[2]//2+2, 1] = 1.0  # modify x, poke sideways to trigger instability
    v0 = StaggeredGrid(unstack_staggered_tensor(vn), st.velocity.box)

st = [ st.copied_with(density=d0, velocity=v0) for _ in range(1) ]  # NOTE: update according to the input feature!

# velocity BC
vn = np.zeros(st[0].velocity.data[0].data.shape)
vn[..., 0:2, 0:vn.shape[2]-1, 0] = 1.0
vn[..., 0:vn.shape[1], 0:1,   0] = 1.0
vn[..., 0:vn.shape[1], -1:,   0] = 1.0
velBCy = vn
velBCyMask = np.copy(vn)        # warning, only works for 1s, otherwise setup/scale

# extra field
cv = st[0].staggered_grid(name="corr", value=0)

# phiflow scene
scene = Scene.create(directory=params['output'])

log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))
log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

if params['output']:
    with open(os.path.normpath(scene.path)+'/params.pickle', 'wb') as f: pickle.dump(params, f)


# load a tf model and stats used for data normalization
with open(params['stats'], 'rb') as f: data_stats = pickle.load(f)
log.info(data_stats)
model = keras.models.load_model(params['model'])  # Fully convolutional, so we can use trained weights regardless the input dimension
model.summary(print_fn=log.info)

scene.write(
    [st[-1].density, st[-1].velocity, cv],
    ['denTf', 'velTf', 'corTf'],
    0
)

simulator = KarmanFlow()
for i in range(1, params['simsteps']):
    for j in range(len(st)-1):
        st[j] = st[j+1]

    st[-1] = simulator.step(st[-1], re=params['re'], res=params['res'], velBCy=velBCy, velBCyMask=velBCyMask)

    inputf = to_feature(st, params['re'])/[
        *(data_stats['std'][1]),    # velocity
        data_stats['ext.std'][0]    # Re
    ]
    cv_pred = model.predict(inputf)*data_stats['std'][1]
    cv = to_staggered(cv_pred, st[-1].velocity.box)
    st[-1] = st[-1].copied_with(velocity=st[-1].velocity + cv)

    log.info('step {:06d}'.format(i))
    scene.write(
        [st[-1].density, st[-1].velocity, cv],
        ['denTf', 'velTf', 'corTf'],
        i
    )
