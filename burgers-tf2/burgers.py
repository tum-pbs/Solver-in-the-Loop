# ----------------------------------------------------------------------------
#
# Phiflow Burgers equation solver framework
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
parser.add_argument('-o', '--output',    default=None,            help='output directory')
parser.add_argument('--thumb',           action='store_true',     help='save thumbnail images')
parser.add_argument('--noforce',         action='store_true',     help='no randomized external forces')
parser.add_argument('-s', '--skipsteps', default=0, type=int,     help='skip first steps')
parser.add_argument('-t', '--simsteps',  default=200, type=int,   help='simulation steps after skipsteps')
parser.add_argument('-r', '--res',       default=32, type=int,    help='resolution of the reference axis')
parser.add_argument('-l', '--len',       default=32, type=int,    help='length of the reference axis')
parser.add_argument('--dt',              default=0.1, type=float, help='simulation time step size')
parser.add_argument('--initvH',          default=None,            help='load hires (will be downsampled) velocity (e.g., velo_0000.npz)')
parser.add_argument('--loadfH',          default=None,            help='load hires (will be downsampled) force files (will be passed to glob) (e.g., "sim_000000/forc_0*.npz")')
parser.add_argument('-d', '--scale',     default=4, type=int,     help='down-sampling scale of hires (only valid when initvH given)')
parser.add_argument('--seed',            default=0, type=int,     help='seed for random number generator')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

if params['cuda']: from phi.tf.tf_cuda_pressuresolver import CUDASolver

from phi.tf.flow import *
from tensorflow import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tf_session = tf.Session(config=config)

random.seed(params['seed'])
np.random.seed(params['seed'])
tf.compat.v1.set_random_seed(params['seed'])

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

class ForcingPhysics(Physics):
    def __init__(self, omega):
        Physics.__init__(self)
        self.omega = omega

    def step(self, effect, dt=1.0, **dependent_states):
        f = effect.field
        f = f.copied_with(phase_offset=f.phase_offset + dt*self.omega, age=f.age + dt)
        return effect.copied_with(field=f, age=effect.age + dt)


batch_size = 1
num_forces = 20
forces = []
force_physics = []
rnd = math.choose_backend([0])
for i in range(num_forces):
    angle = rnd.random_uniform([batch_size, 1, 1, 1]) * np.pi
    unit_direction = math.concat([math.sin(angle), math.cos(angle)], axis=-1)
    wave_vec_i = (rnd.random_uniform([batch_size, 1, 1, 1]) + 1) * 0.8 * unit_direction
    amplitude = (rnd.random_uniform([batch_size, 1, 1, 2]) - 0.5) * 0.3
    phase_offset = rnd.random_uniform([batch_size]) * 2 * np.pi
    omega = rnd.random_uniform([batch_size]) * 0.8 - 0.4
    force = FieldEffect(SinPotential(wave_vec_i, phase_offset=phase_offset, data=amplitude), ['velocity'])
    forces.append(force)
    force_physics.append(ForcingPhysics(omega))

# init velocity

fc_files = sorted(glob.glob(params['loadfH'])) if params['loadfH'] else None

dm = Domain([params['res']]*2, box=box([params['len']]*2), boundaries=PERIODIC)
st = BurgersVelocitySMAC(dm, velocity=lambda s: math.randfreq(s) * 2)
fc = st.copied_with(velocity=sum([force.field for force in forces]).at(dm.staggered_grid(0)))

if params['initvH']: st = st.copied_with(velocity=downVel(read_zipped_array(params['initvH'])))
if fc_files:
    fc = fc.copied_with(velocity=downVel(read_zipped_array(fc_files[0])))
    log.info('Will load force files: {}'.format(fc_files))

# phiflow scene

scene = Scene.create(directory=params['output'])
sess  = Session(scene, session=tf_session)

log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))
log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

if params['output']:
    with open(os.path.normpath(scene.path)+'/params.pickle', 'wb') as f: pickle.dump(params, f)

simulator = BurgersTest()

tf_st_in = placeholder(st.shape)
tf_fc_in = placeholder(fc.shape)
tf_st = simulator.step_with_f(v=tf_st_in, f=tf_fc_in, dt=params['dt']) if not params['noforce'] else simulator.step(v=tf_st_in, dt=params['dt'])
tf_fc = tf_fc_in

if fc_files is None:            # for regular sim with randomization forces
    tf_fcs_in = placeholder_like(forces)
    tf_fcs = [ physics.step(force, dt=params['dt']) for force, physics in zip(tf_fcs_in, force_physics) ]

if params['skipsteps']==0 and params['output'] is not None:
    scene.write(
        [st.velocity, fc.velocity],
        ['velo', 'forc'],
        0
    )
    if params['thumb']:
        thumb_path = os.path.normpath(scene.path).replace(os.path.basename(scene.path), "thumb/{}".format(os.path.basename(scene.path)))
        distutils.dir_util.mkpath(thumb_path)
        save_img(st.velocity.data[1].data, 100000., thumb_path + "/velU_{:06d}.png".format(0))
        save_img(st.velocity.data[0].data, 100000., thumb_path + "/velV_{:06d}.png".format(0))
        save_img(fc.velocity.data[1].data, 100000., thumb_path + "/frcU_{:06d}.png".format(0))
        save_img(fc.velocity.data[0].data, 100000., thumb_path + "/frcV_{:06d}.png".format(0))

sess.initialize_variables()
for i in range(1, max(params['simsteps']+params['skipsteps'], 1)):
    my_feed_dict = { tf_st_in: st, tf_fc_in: fc }
    st, fc = sess.run([tf_st, tf_fc], my_feed_dict)

    if fc_files is None:
        forces = sess.run(tf_fcs, {atf_fc_in: force for atf_fc_in, force in zip(tf_fcs_in, forces)})
        fc = fc.copied_with(velocity=sum([force.field for force in forces]).at(dm.staggered_grid(0)))

    else:
        fc = fc.copied_with(velocity=downVel(read_zipped_array(fc_files[i])))

    log.info('Step {:06d}'.format(i))
    if params['skipsteps']<=i and params['output'] is not None:
        scene.write(
            [st.velocity, fc.velocity],
            ['velo', 'forc'],
            max(i - params['skipsteps'], 0)
        )
        if params['thumb'] and params['output'] is not None:
            thumb_path = os.path.normpath(scene.path).replace(os.path.basename(scene.path), "thumb/{}".format(os.path.basename(scene.path)))
            distutils.dir_util.mkpath(thumb_path)
            save_img(st.velocity.data[1].data, 100000., thumb_path + "/velU_{:06d}.png".format(max(i - params['skipsteps'], 0)))
            save_img(st.velocity.data[0].data, 100000., thumb_path + "/velV_{:06d}.png".format(max(i - params['skipsteps'], 0)))
            save_img(fc.velocity.data[1].data, 100000., thumb_path + "/frcU_{:06d}.png".format(max(i - params['skipsteps'], 0)))
            save_img(fc.velocity.data[0].data, 100000., thumb_path + "/frcV_{:06d}.png".format(max(i - params['skipsteps'], 0)))
