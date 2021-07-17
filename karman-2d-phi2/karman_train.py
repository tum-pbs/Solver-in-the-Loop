# ----------------------------------------------------------------------------
#
# Phiflow Karman vortex solver framework
# Copyright 2020-2021 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Training
#
# ----------------------------------------------------------------------------

import os, sys, logging, argparse, pickle, glob, random, distutils.dir_util

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',            default='0',               help='visible GPUs')
parser.add_argument('--train',          default=None,              help='training; will load data from this simulation folder (set) and save down-sampled files')
parser.add_argument('--skip-ds',        action='store_true',       help='skip down-scaling; assume you have already saved')
parser.add_argument('--only-ds',        action='store_true',       help='exit after down-scaling and saving; use only for data pre-processing')
parser.add_argument('--log',            default=None,              help='path to a log file')
parser.add_argument('-s', '--scale',    default=4, type=int,       help='simulation scale for high-res')
parser.add_argument('-n', '--nsims',    default=1, type=int,       help='number of simulations')
parser.add_argument('-b', '--sbatch',   default=1, type=int,       help='size of a batch; when 10 simulations with the size of 5, 5 simulations are into two batches')
parser.add_argument('-t', '--simsteps', default=1500, type=int,    help='simulation steps; # of data samples (i.e. frames) per simulation')
parser.add_argument('-m', '--msteps',   default=2, type=int,       help='multi steps in training loss')
parser.add_argument('-e', '--epochs',   default=10, type=int,      help='training epochs')
parser.add_argument('--seed',           default=None, type=int,    help='seed for random number generator')
parser.add_argument('-r', '--res',      default=32, type=int,      help='target (i.e., low-res) resolution') # FIXME: save and restore from the data
parser.add_argument('-l', '--len',      default=100, type=int,     help='length of the reference axis')      # FIXME: save and restore from the data
parser.add_argument('--model',          default='mars_moon',       help='(predefined) network model')
parser.add_argument('--reg-loss',       action='store_true',       help='turn on regularization loss')
parser.add_argument('--lr',             default=1e-3, type=float,  help='start learning rate')
parser.add_argument('--adplr',          action='store_true',       help='turn on adaptive learning rate')
parser.add_argument('--clip-grad',      action='store_true',       help='turn on clip gradients')
parser.add_argument('--resume',         default=-1, type=int,      help='resume training epochs')
parser.add_argument('--inittf',         default=None,              help='load initial model weights (warm start)')
parser.add_argument('--pretf',          default=None,              help='load pre-trained weights (only for testing pre-trained supervised model; do not use for a warm start!)')
parser.add_argument('--tf',             default='/tmp/phiflow/tf', help='path to a tensorflow output dir (model, logs, etc.)')
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

if params['resume']>0 and params['log']:
    params['log'] = os.path.splitext(params['log'])[0] + '_resume{:04d}'.format(params['resume']) + os.path.splitext(params['log'])[1]

if params['log']:
    distutils.dir_util.mkpath(os.path.dirname(params['log']))
    log.addHandler(logging.FileHandler(params['log']))

if (params['nsims'] % params['sbatch']) != 0:
    params['nsims'] = (params['nsims']//params['sbatch'])*params['sbatch']
    log.info('Number of simulations is not divided by the batch size thus adjusted to {}'.format(params['nsims']))

log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

def model_mercury(inputs_dict):
    with tf.name_scope('model_mercury') as scope:
        return keras.Sequential([
            keras.layers.Input(**inputs_dict),
            keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu),
            keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation=tf.nn.relu),
            keras.layers.Conv2D(filters=2,  kernel_size=5, padding='same', activation=None), # u, v
        ], name='mercury')

def model_mars_moon(inputs_dict):
    with tf.name_scope('model_mars_moon') as scope:
        l_input = keras.layers.Input(**inputs_dict)
        block_0 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(l_input)
        block_0 = keras.layers.LeakyReLU()(block_0)

        l_conv1 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(block_0)
        l_conv1 = keras.layers.LeakyReLU()(l_conv1)
        l_conv2 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(l_conv1)
        l_skip1 = keras.layers.add([block_0, l_conv2])
        block_1 = keras.layers.LeakyReLU()(l_skip1)

        l_conv3 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(block_1)
        l_conv3 = keras.layers.LeakyReLU()(l_conv3)
        l_conv4 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(l_conv3)
        l_skip2 = keras.layers.add([block_1, l_conv4])
        block_2 = keras.layers.LeakyReLU()(l_skip2)

        l_conv5 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(block_2)
        l_conv5 = keras.layers.LeakyReLU()(l_conv5)
        l_conv6 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(l_conv5)
        l_skip3 = keras.layers.add([block_2, l_conv6])
        block_3 = keras.layers.LeakyReLU()(l_skip3)

        l_conv7 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(block_3)
        l_conv7 = keras.layers.LeakyReLU()(l_conv7)
        l_conv8 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(l_conv7)
        l_skip4 = keras.layers.add([block_3, l_conv8])
        block_4 = keras.layers.LeakyReLU()(l_skip4)

        l_conv9 = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(block_4)
        l_conv9 = keras.layers.LeakyReLU()(l_conv9)
        l_convA = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(l_conv9)
        l_skip5 = keras.layers.add([block_4, l_convA])
        block_5 = keras.layers.LeakyReLU()(l_skip5)

        l_output = keras.layers.Conv2D(filters=2,  kernel_size=5, padding='same')(block_5)
        return keras.models.Model(inputs=l_input, outputs=l_output, name='mars_moon')

def lr_schedule(epoch, current_lr):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 10, 15, 20, 22 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = current_lr
    if   epoch == 23: lr *= 0.5
    elif epoch == 21: lr *= 1e-1
    elif epoch == 16: lr *= 1e-1
    elif epoch == 11: lr *= 1e-1
    return lr


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

    def step(self, density_in, velocity_in, re, res, buoyancy_factor=0, dt=1.0, make_input_divfree=False, make_output_divfree=True): #, conserve_density=True):
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

class PhifDataset():
    def __init__(self, domain, dirpath, num_frames, num_sims=None, batch_size=1, print_fn=print, skip_preprocessing=False):
        self.dataSims      = sorted(glob.glob(dirpath + '/sim_0*'))[0:num_sims]
        self.pathsDen      = [ sorted(glob.glob(asim + '/dens_0*.npz')) for asim in self.dataSims ]
        self.pathsVel      = [ sorted(glob.glob(asim + '/velo_0*.npz')) for asim in self.dataSims ]
        self.dataFrms      = [ np.arange(num_frames) for _ in self.dataSims ] # NOTE: may contain different numbers of frames
        self.batchSize     = batch_size
        self.epoch         = None
        self.epochIdx      = 0
        self.batch         = None
        self.batchIdx      = 0
        self.step          = None
        self.stepIdx       = 0
        self.dataPreloaded = None
        self.printFn       = print_fn
        self.domain        = domain # phiflow: target domain (i.e., low-res.)

        self.numOfSims    = num_sims
        self.numOfBatchs  = self.numOfSims//self.batchSize
        self.numOfFrames  = num_frames
        self.numOfSteps   = num_frames

        if not skip_preprocessing:
            self.printFn('Pre-processing: Loading data from {} = {} and save down-scaled data'.format(dirpath, self.dataSims))
            for j,asim in enumerate(self.dataSims):
                for i in range(num_frames):
                    if not os.path.isfile(self.filenameToDownscaled(self.pathsDen[j][i])):
                        d = phi.field.read(file=self.pathsDen[j][i]).at(self.domain.scalar_grid())
                        phi.field.write(field=d, file=self.filenameToDownscaled(self.pathsDen[j][i]))
                        self.printFn('Wrote {}'.format(self.filenameToDownscaled(self.pathsDen[j][i])))
                    if not os.path.isfile(self.filenameToDownscaled(self.pathsVel[j][i])):
                        v = phi.field.read(file=self.pathsVel[j][i]).at(self.domain.staggered_grid())
                        phi.field.write(field=v, file=self.filenameToDownscaled(self.pathsVel[j][i]))
                        self.printFn('Wrote {}'.format(self.filenameToDownscaled(self.pathsVel[j][i])))

        self.printFn('Preload: Loading data from {} = {}'.format(dirpath, self.dataSims))
        self.dataPreloaded = {  # dataPreloaded['sim_key'][frame number][0=density, 1=x-velocity, 2=y-velocity]
            asim: [
                (
                    np.expand_dims(phi.field.read(file=self.filenameToDownscaled(self.pathsDen[j][i])).values.numpy(('y', 'x')),             axis=0), # density
                    np.expand_dims(phi.field.read(file=self.filenameToDownscaled(self.pathsVel[j][i])).vector['x'].values.numpy(('y', 'x')), axis=0), # x-velocity
                    np.expand_dims(phi.field.read(file=self.filenameToDownscaled(self.pathsVel[j][i])).vector['y'].values.numpy(('y', 'x')), axis=0), # y-velocity
                ) for i in range(num_frames)
            ] for j,asim in enumerate(self.dataSims)
        }                       # for each, keep shape=[batch-size, res-y, res-x]
        assert len(self.dataPreloaded[self.dataSims[0]][0][0].shape)==3, 'Data shape is wrong.'
        assert len(self.dataPreloaded[self.dataSims[0]][0][1].shape)==3, 'Data shape is wrong.'
        assert len(self.dataPreloaded[self.dataSims[0]][0][2].shape)==3, 'Data shape is wrong.'

        self.dataStats = {
            'std': (
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][0].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)), # density
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][1].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)), # x-velocity
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][2].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)), # y-velocity
            )
        }

        self.extConstChannelPerSim = {} # extConstChannelPerSim['sim_key'][0=first channel, ...]; for now, only Reynolds Nr.
        num_of_ext_channel = 1
        for asim in self.dataSims:
            with open(asim+'/params.pickle', 'rb') as f:
                sim_params = pickle.load(f)
                self.extConstChannelPerSim[asim] = [ sim_params['re'] ] # Reynolds Nr.

        self.dataStats.update({
            'ext.std': [
                np.std([np.absolute(self.extConstChannelPerSim[asim][i]) for asim in self.dataSims]) for i in range(num_of_ext_channel) # Reynolds Nr
            ]
        })
        self.printFn(self.dataStats)

    def filenameToDownscaled(self, fname):
        return os.path.dirname(fname) + '/ds_' + os.path.basename(fname)

    def getInstance(self, sim_idx=0, frame=0):
        d0_hi = math.concat([self.dataPreloaded[self.dataSims[sim_idx+i]][frame][0] for i in range(self.batchSize)], axis=0)
        u0_hi = math.concat([self.dataPreloaded[self.dataSims[sim_idx+i]][frame][1] for i in range(self.batchSize)], axis=0)
        v0_hi = math.concat([self.dataPreloaded[self.dataSims[sim_idx+i]][frame][2] for i in range(self.batchSize)], axis=0)
        return [d0_hi, u0_hi, v0_hi] # TODO: additional channels

    def newEpoch(self, exclude_tail=0, shuffle_data=True):
        self.numOfSteps = self.numOfFrames - exclude_tail
        sim_frames = [ (asim, self.dataFrms[i][0:(len(self.dataFrms[i])-exclude_tail)]) for i,asim in enumerate(self.dataSims) ]
        sim_frame_pairs = []
        for i,_ in enumerate(sim_frames):
            sim_frame_pairs += [ (i, aframe) for aframe in sim_frames[i][1] ] # [(sim_idx, frame_number), ...]

        if shuffle_data: random.shuffle(sim_frame_pairs)
        self.epoch = [ list(sim_frame_pairs[i*self.numOfSteps:(i+1)*self.numOfSteps]) for i in range(self.batchSize*self.numOfBatchs) ]
        self.epochIdx += 1
        self.batchIdx = 0
        self.stepIdx = 0

    def nextBatch(self):        # batch size may be the number of simulations in a batch
        self.batchIdx += self.batchSize
        self.stepIdx = 0

    def nextStep(self):
        self.stepIdx += 1

    def getData(self, consecutive_frames, with_skip=1):
        d_hi = [
            np.concatenate([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]] # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip # frames
                ][0]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        u_hi = [
            np.concatenate([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]] # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip # frames
                ][1]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        v_hi = [
            np.concatenate([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]] # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip # frames
                ][2]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        ext = [
            self.extConstChannelPerSim[
                self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
            ][0] for i in range(self.batchSize)
        ]
        return [d_hi, u_hi, v_hi, ext]

    def getPrevData(self, previous_frames, with_skip=1): # NOTE: not in use; need to test
        d_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
                ][
                    max([0, self.epoch[self.batchIdx+i][self.stepIdx][1]-j*with_skip])
                ][0]
                for i in range(self.batchSize)
            ], axis=0) for j in range(previous_frames)
        ]
        u_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
                ][
                    max([0, self.epoch[self.batchIdx+i][self.stepIdx][1]-j*with_skip])
                ][1]
                for i in range(self.batchSize)
            ], axis=0) for j in range(previous_frames)
        ]
        v_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
                ][
                    max([0, self.epoch[self.batchIdx+i][self.stepIdx][1]-j*with_skip])
                ][2]
                for i in range(self.batchSize)
            ], axis=0) for j in range(previous_frames)
        ]
        # TODO: additional channels
        return [d_hi, v_hi]


domain  = Domain(y=params['res']*2, x=params['res'], bounds=Box[0:params['len']*2, 0:params['len']], boundaries=OPEN)
simulator_lo = KarmanFlow(domain=domain)

dataset = PhifDataset(
    domain=domain,
    dirpath=params['train'],
    num_frames=params['simsteps'], num_sims=params['nsims'], batch_size=params['sbatch'],
    print_fn=log.info,
    skip_preprocessing=params['skip_ds']
)
if params['only_ds']: exit(0)

if params['pretf']:
    with open(os.path.dirname(params['pretf'])+'/stats.pickle', 'rb') as f: ld_stats = pickle.load(f)
    dataset.dataStats['in.std'] = (ld_stats['in.std'][0], (ld_stats['in.std'][1], ld_stats['in.std'][2]))
    dataset.dataStats['out.std'] = ld_stats['out.std']
    log.info(dataset.dataStats)

if params['resume']>0:
    with open(params['tf']+'/dataStats.pickle', 'rb') as f: dataset.dataStats = pickle.load(f)

if (params['train'] is None):
    log.info(params['train'])
    log.info('No pre-loadable training data path is given.')
    exit(0)

tf_tb_writer = tf.summary.create_file_writer(params['tf']+'/summary/training')

# model
netModel = eval('model_{}'.format(params['model']))
model = netModel(dict(shape=(params['res']*2, params['res'], 3)))
model.summary(print_fn=log.info)

if params['pretf']:
    log.info('load a pre-trained model: {}'.format(params['pretf']))
    ld_model = keras.models.load_model(params['pretf'], compile=False)
    model.set_weights(ld_model.get_weights())

if params['inittf']:
    log.info('load an initial model (warm start): {}'.format(params['inittf']))
    ld_model = keras.models.load_model(params['inittf'], compile=False)
    model.set_weights(ld_model.get_weights())

if params['resume']<1:
    [ params['tf'] and distutils.dir_util.mkpath(params['tf']) ]
    with open(params['tf']+'/dataStats.pickle', 'wb') as f: pickle.dump(dataset.dataStats, f)

else:
    ld_model = keras.models.load_model(params['tf']+'/model_epoch{:04d}.h5'.format(params['resume']))
    model.set_weights(ld_model.get_weights())

opt = tf.keras.optimizers.Adam(learning_rate=params['lr'])

def to_feature(dens_vel_grid_array, ext_const_channel):
    # drop the unused edges of the staggered velocity grid making its dim same to the centered grid's
    with tf.name_scope('to_feature') as scope:
        return math.stack(
            [
                dens_vel_grid_array[1].vector['x'].x[:-1].values,         # u
                dens_vel_grid_array[1].vector['y'].y[:-1].values,         # v
                math.ones(dens_vel_grid_array[0].shape)*ext_const_channel # Re
            ],
            math.channel('channels')
        )

def to_staggered(tf_tensor, domain):
    with tf.name_scope('to_staggered') as scope:
        return domain.staggered_grid(
            math.stack(
                [
                    math.tensor(tf.pad(tf_tensor[..., 1], [(0,0), (0,1), (0,0)]), math.batch('batch'), math.spatial('y, x')), # v
                    math.tensor(tf.pad(tf_tensor[..., 0], [(0,0), (0,0), (0,1)]), math.batch('batch'), math.spatial('y, x')), # u
                ], math.channel('vector')
            )
        )

def train_step(pf_in_dens_gt, pf_in_velo_gt, pf_in_Re, i_step):
    with tf.name_scope('train_step'), tf.GradientTape() as tape:
        with tf.name_scope('sol') as scope:
            pf_co_prd, pf_cv_md = [], [] # predicted states with correction, inferred velocity corrections
            for i in range(params['msteps']):
                with tf.name_scope('solve_and_correct') as scope:
                    with tf.name_scope('solver_step') as scope:
                        pf_co_prd += [
                            simulator_lo.step(
                                density_in=pf_in_dens_gt[0] if i==0 else pf_co_prd[-1][0],
                                velocity_in=pf_in_velo_gt[0] if i==0 else pf_co_prd[-1][1],
                                re=pf_in_Re,
                                res=params['res'],
                            )
                        ]       # pf_co_prd: [[density1, velocity1], [density2, velocity2], ...]

                    with tf.name_scope('pred') as scope:
                        model_input = to_feature(pf_co_prd[-1], pf_in_Re)
                        model_input /= math.tensor([dataset.dataStats['std'][1], dataset.dataStats['std'][2], dataset.dataStats['ext.std'][0]], channel('channels')) # [u, v, Re]
                        model_out = model(model_input.native(['batch', 'y', 'x', 'channels']), training=True)
                        model_out *= [dataset.dataStats['std'][1], dataset.dataStats['std'][2]] # [u, v]
                        pf_cv_md += [ to_staggered(model_out, domain) ]                         # pf_cv_md: [velocity_correction1, velocity_correction2, ...]

                    pf_co_prd[-1][1] = pf_co_prd[-1][1] + pf_cv_md[-1]

        with tf.name_scope('loss') as scope, tf_tb_writer.as_default():
            with tf.name_scope('steps_x') as scope:
                loss_steps_x = [
                    tf.nn.l2_loss(
                        (
                            pf_in_velo_gt[i+1].vector['x'].values.native(('batch', 'y', 'x'))
                            - pf_co_prd[i][1].vector['x'].values.native(('batch', 'y', 'x'))
                        )/dataset.dataStats['std'][1]
                    )
                    for i in range(params['msteps'])
                ]
                loss_steps_x_sum = tf.math.reduce_sum(loss_steps_x)

            with tf.name_scope('steps_y') as scope:
                loss_steps_y = [
                    tf.nn.l2_loss(
                        (
                            pf_in_velo_gt[i+1].vector['y'].values.native(('batch', 'y', 'x'))
                            - pf_co_prd[i][1].vector['y'].values.native(('batch', 'y', 'x'))
                        )/dataset.dataStats['std'][2]
                    )
                    for i in range(params['msteps'])
                ]
                loss_steps_y_sum = tf.math.reduce_sum(loss_steps_y)

            loss = (loss_steps_x_sum + loss_steps_y_sum)/params['msteps']

            for i,a_step_loss in enumerate(loss_steps_x): tf.summary.scalar(name='loss_each_step_vel_x{:02d}'.format(i+1), data=a_step_loss, step=math.to_int64(i_step).native())
            for i,a_step_loss in enumerate(loss_steps_y): tf.summary.scalar(name='loss_each_step_vel_y{:02d}'.format(i+1), data=a_step_loss, step=math.to_int64(i_step).native())
            tf.summary.scalar(name='sum_steps_loss', data=loss, step=math.to_int64(i_step).native())

            total_loss = loss
            if params['reg_loss']:
                reg_loss = tf.math.add_n(model.losses)
                total_loss += reg_loss
                tf.summary.scalar(name='loss_regularization', data=reg_loss, step=math.to_int64(i_step).native())

            tf.summary.scalar(name='loss', data=total_loss, step=math.to_int64(i_step).native())

        with tf.name_scope('apply_gradients') as scope:
            gradients = tape.gradient(total_loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))

        return math.tensor(total_loss)

jit_step = math.jit_compile(train_step)

i_st = 0
for j in range(params['epochs']): # training
    dataset.newEpoch(exclude_tail=params['msteps'])
    if j<params['resume']:
        log.info('resume: skipping {} epoch'.format(j+1))
        i_st += dataset.numOfSteps*dataset.numOfBatchs
        continue

    for ib in range(dataset.numOfBatchs):   # for each batch
        for i in range(dataset.numOfSteps): # for each step
            # adata: [[dens0, dens1, ...], [x-velo0, x-velo1, ...], [y-velo0, y-velo1, ...], [ReynoldsNr(s)]]
            adata = dataset.getData(consecutive_frames=params['msteps'], with_skip=1)
            dens_gt = [         # [density0:CenteredGrid, density1, ...]
                domain.scalar_grid(
                    math.tensor(adata[0][k], math.batch('batch'), math.spatial('y, x'))
                ) for k in range(params['msteps']+1)
            ]
            velo_gt = [         # [velocity0:StaggeredGrid, velocity1, ...]
                domain.staggered_grid(
                    math.stack(
                        [
                            math.tensor(adata[2][k], math.batch('batch'), math.spatial('y, x')),
                            math.tensor(adata[1][k], math.batch('batch'), math.spatial('y, x')),
                        ], math.channel('vector')
                    )
                ) for k in range(params['msteps']+1)
            ]
            re_nr = math.tensor(adata[3], math.batch('batch'))

            if i_st==0: tf.summary.trace_on(graph=True, profiler=True)

            l2 = jit_step(dens_gt, velo_gt, re_nr, math.tensor(i_st))

            if i_st==0:
                with tf_tb_writer.as_default():
                    tf.summary.trace_export(name="trace_train_step", step=i_st, profiler_outdir=params['tf']+'/summary/training')

            i_st += 1

            log.info('epoch {:03d}/{:03d}, batch {:03d}/{:03d}, step {:04d}/{:04d}: loss={}'.format(
                j+1, params['epochs'], ib+1, dataset.numOfBatchs, i+1, dataset.numOfSteps, l2
            ))
            dataset.nextStep()

        dataset.nextBatch()

    if j%10==9: model.save(params['tf']+'/model_epoch{:04d}.h5'.format(j+1))

tf_tb_writer.close()
model.save(params['tf']+'/model.h5')
