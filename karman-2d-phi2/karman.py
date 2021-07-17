# ----------------------------------------------------------------------------
#
# Phiflow Karman vortex solver framework
# Copyright 2020-2021 Kiwon Um, Nils Thuerey
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
    assert len(array.shape) == 2, 'cannot save as an image of {}'.format(array.shape)
    ima = np.reshape(array, [array.shape[0], array.shape[1]]) # remove channel dimension, 2d
    # ima = ima[::-1, :]  # flip along y
    image = Image.fromarray(np.asarray(ima * scale, dtype='i'))
    print("\tWriting image: " + name)
    image.save(name)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',             default='0',             help='visible GPUs')
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

from phi.physics._boundaries import Domain, OPEN, STICKY as CLOSED
from phi.tf.flow import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    log.info('{} Physical GPUs {} Logical GPUs'.format(len(gpus), len(logical_gpus)))

from tensorflow import keras

random.seed(params['seed'])
np.random.seed(params['seed'])
tf.random.set_seed(params['seed'])

class KarmanFlow():
    def __init__(self, domain):
        self.domain = domain

        shape_v = self.domain.staggered_grid(0).vector['y'].shape
        vel_yBc = np.zeros(shape_v.sizes)
        vel_yBc[0:2, 0:vel_yBc.shape[1]-1] = 1.0
        vel_yBc[0:vel_yBc.shape[0], 0:1] = 1.0
        vel_yBc[0:vel_yBc.shape[0], -1:] = 1.0
        self.vel_yBc = math.tensor(vel_yBc, shape_v)
        self.vel_yBcMask = math.tensor(np.copy(vel_yBc), shape_v) # warning, only works for 1s, otherwise setup/scale

        self.inflow = self.domain.scalar_grid(Box[5:10, 25:75])         # TODO: scale with domain if necessary!
        self.obstacles = [Obstacle(Sphere(center=[50, 50], radius=10))] # TODO: scale with domain if necessary!

    def step(self, density_in, velocity_in, re, res, dt=1.0, make_input_divfree=False, make_output_divfree=True): #, conserve_density=True):
        velocity = velocity_in
        density = density_in

        # apply viscosity
        velocity = phi.flow.diffuse.explicit(field=velocity, diffusivity=1.0/re*dt*res*res, dt=dt)
        vel_x = velocity.vector['x']
        vel_y = velocity.vector['y']

        # apply velocity BCs, only y for now; velBCy should be pre-multiplied
        vel_y = vel_y*(1.0 - self.vel_yBcMask) + self.vel_yBc
        velocity = self.domain.staggered_grid(phi.math.stack([vel_y.data, vel_x.data], channel('vector')))

        pressure = None
        if make_input_divfree:
            velocity, pressure = fluid.make_incompressible(velocity, self.obstacles)

        # --- Advection ---
        density = advect.semi_lagrangian(density+self.inflow, velocity, dt=dt)
        velocity = advected_velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
        # if conserve_density and self.domain.boundaries['accessible_extrapolation'] == math.extrapolation.ZERO:  # solid boundary
        #     density = field.normalize(density, self.density)

        # --- Pressure solve ---
        if make_output_divfree:
            velocity, pressure = fluid.make_incompressible(velocity, self.obstacles)

        self.solve_info = {
            'pressure': pressure,
            'advected_velocity': advected_velocity,
        }

        return [density, velocity]


scene = Scene.create(parent_directory=params['output']) # phiflow scene

log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))
log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

if params['output']:
    with open(os.path.normpath(scene.path)+'/params.pickle', 'wb') as f: pickle.dump(params, f)

domain = Domain(y=params['res']*2, x=params['res'], bounds=Box[0:params['len']*2, 0:params['len']], boundaries=OPEN)

# init density & velocity
d0 = phi.field.read(params['initdH']).at(domain.scalar_grid()) if params['initdH'] else domain.scalar_grid(0)

if params['initvH']:
    v0 = phi.field.read(params['initvH']).at(domain.staggered_grid())

else:
    vv = np.ones(domain.staggered_grid().vector['y'].shape.sizes) # warm start - initialize flow to 1 along y everywhere
    uu = np.zeros(domain.staggered_grid().vector['x'].shape.sizes)
    uu[uu.shape[0]//2+10:uu.shape[0]//2+20, uu.shape[1]//2-2:uu.shape[1]//2+2] = 1.0 # modify x, poke sideways to trigger instability
    v0 = domain.staggered_grid(math.stack([math.tensor(vv, spatial('y, x')), math.tensor(uu, spatial('y, x'))], channel('vector')))

simulator = KarmanFlow(domain=domain)
density, velocity = d0, v0

jit_step = math.jit_compile(simulator.step)

if params['skipsteps']==0 and params['output']:
    scene.write(
        data = {
            'dens': density,
            'velo': velocity,
        },
        frame=0
    )

    if params['thumb']:
        thumb_path = os.path.normpath(scene.path).replace(os.path.basename(scene.path), "thumb/{}".format(os.path.basename(scene.path)))
        distutils.dir_util.mkpath(thumb_path)
        save_img(density.data.numpy(density.values.shape.names),    10000., thumb_path + "/dens_{:06d}.png".format(0))                        # shape: [cy, cx]
        save_img(velocity.vector['x'].data.numpy(velocity.vector['x'].values.shape.names), 40000., thumb_path + "/velU_{:06d}.png".format(0)) # shape: [cy, cx+1]
        save_img(velocity.vector['y'].data.numpy(velocity.vector['y'].values.shape.names), 40000., thumb_path + "/velV_{:06d}.png".format(0)) # shape: [cy+1, cx]

for i in range(1, params['simsteps']):
    log.info('Step {:06d}'.format(i))
    density, velocity = jit_step(
        density,
        velocity,
        re=params['re'],
        res=params['res']
    )

    if params['skipsteps']<i and params['output']:
        scene.write(
            data = {
                'dens': density,
                'velo': velocity,
            },
            frame=i
        )

        if params['thumb']:
            thumb_path = os.path.normpath(scene.path).replace(os.path.basename(scene.path), "thumb/{}".format(os.path.basename(scene.path)))
            distutils.dir_util.mkpath(thumb_path)
            save_img(density.data.numpy(density.values.shape.names),    10000., thumb_path + "/dens_{:06d}.png".format(i))                        # shape: [cy, cx]
            save_img(velocity.vector['x'].data.numpy(velocity.vector['x'].values.shape.names), 40000., thumb_path + "/velU_{:06d}.png".format(i)) # shape: [cy, cx+1]
            save_img(velocity.vector['y'].data.numpy(velocity.vector['y'].values.shape.names), 40000., thumb_path + "/velV_{:06d}.png".format(i)) # shape: [cy+1, cx]
