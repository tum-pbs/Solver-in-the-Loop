# ----------------------------------------------------------------------------
#
# Phiflow Burgers equation solver framework
# Copyright 2020 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Apply correction network model
#
# ----------------------------------------------------------------------------

import os, sys, glob, logging, argparse, pickle

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',            default='0',                                  help='visible GPUs')
parser.add_argument('-t', '--simsteps', default=200, type=int,                        help='simulation steps')
parser.add_argument('-r', '--res',      default=32, type=int,                         help='resolution of the reference axis')
parser.add_argument('-l', '--len',      default=96, type=int,                         help='length of the reference axis')
parser.add_argument('--dt',             default=1.0, type=float,                      help='simulation time step size')
parser.add_argument('--noforce',        action='store_true',                          help='no randomized external forces')
parser.add_argument('--initvH',         default=None,                                 help='load hires (will be downsampled) velocity (e.g., velo_0000.npz)')
parser.add_argument('--loadfH',         default=None,                                 help='load hires (will be downsampled) force files (will be passed to glob) (e.g., "sim_000000/forc_0*.npz")')
parser.add_argument('-s', '--scale',    default=4, type=int,                          help='simulation scale for high-res')
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

def to_feature(smokestates, forcestates):
    # input feature used for supervised version; drop the unused edges of the
    # staggered velocity grid making its dim same to the centered grid's
    with tf.name_scope('to_feature') as scope:
        return math.concat(
            [smokestates[j].velocity.staggered_tensor()[:, :-1:, :-1:, 0:2] for j in range(len(smokestates))] +
            [forcestates[j].velocity.staggered_tensor()[:, :-1:, :-1:, 0:2] for j in range(len(forcestates))],
            axis=-1
        )

def to_feature_noforce(smokestates):
    # input feature used for supervised version; drop the unused edges of the
    # staggered velocity grid making its dim same to the centered grid's
    with tf.name_scope('to_feature') as scope:
        return math.concat(
            [smokestates[j].velocity.staggered_tensor()[:, :-1:, :-1:, 0:2] for j in range(len(smokestates))],
            axis=-1
        )

def to_staggered(tensor_cen, box):
    with tf.name_scope('to_staggered') as scope:
        return StaggeredGrid(math.pad(tensor_cen, ((0,0), (0,1), (0,1), (0,0))), box=box)

def downsample4xSMAC(tensor):
    return StaggeredGrid(tensor).downsample2x().downsample2x().staggered_tensor()

downVel = eval('downsample{}xSMAC'.format(params['scale']))

@struct.definition()
class BurgersVelocitySMAC(BurgersVelocity):
    @struct.variable(dependencies=DomainState.domain, default=0)
    def velocity(self, velocity):
        return self.staggered_grid('velocity', velocity)

class BurgersTest(Burgers):
    def __init__(self, default_viscosity=0.1, viscosity=None, diffusion_substeps=1):
        Burgers.__init__(self, default_viscosity=default_viscosity, viscosity=viscosity, diffusion_substeps=diffusion_substeps)

    def step(self, v, dt=1.0, effects=()):
        return super().step(v=v, dt=dt, effects=effects)

    def step_with_f(self, v, f, dt=1.0):
        v_new = super().step(v=v, dt=dt)
        return v_new.copied_with(velocity=v_new.velocity + dt*f.velocity)


# initialize

dm = Domain(resolution=[params['res']]*2, box=box([params['len']]*2), boundaries=PERIODIC)
st = [ BurgersVelocitySMAC(dm, velocity=lambda s: math.randfreq(s) * 2) for _ in range(1) ]  # NOTE: update according to the input feature!

if not params['noforce']:
    fc_files = sorted(glob.glob(params['loadfH'])) if params['loadfH'] else None
    fc = st[0].copied_with(velocity=downVel(read_zipped_array(fc_files[0])))

if params['initvH']:
    st = [ st[0].copied_with(velocity=downVel(read_zipped_array(params['initvH']))) for _ in range(1) ]  # NOTE: update according to the input feature!

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
    [st[-1].velocity, cv],
    ['velTf', 'corTf'],
    0
)

simulator = BurgersTest()
for i in range(1, params['simsteps']):

    for j in range(len(st)-1): st[j] = st[j+1]
    if not params['noforce']:
        st[-1] = simulator.step_with_f(v=st[-1], f=fc, dt=params['dt'])
        fc = fc.copied_with(velocity=downVel(read_zipped_array(fc_files[i])))
        inputf = to_feature(st, [fc,])/[*(data_stats['std'][0]), *(data_stats['std'][1])]

    else:
        st[-1] = simulator.step(v=st[-1], dt=params['dt'])
        inputf = to_feature_noforce(st)/[*(data_stats['std'][0])]

    cv_pred = model.predict(inputf)*data_stats['std'][0]
    cv = to_staggered(cv_pred, st[-1].velocity.box)
    st[-1] = st[-1].copied_with(velocity=st[-1].velocity + cv)

    log.info('step {:06d}'.format(i))
    scene.write(
        [st[-1].velocity, cv],
        ['velTf', 'corTf'],
        i
    )
