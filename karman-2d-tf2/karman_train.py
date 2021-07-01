# ----------------------------------------------------------------------------
#
# Phiflow Karman vortex solver framework
# Copyright 2020 Kiwon Um, Nils Thuerey
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

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',            default='0',               help='visible GPUs')
parser.add_argument('--cuda',           action='store_true',       help='enable CUDA for solver')
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
parser.add_argument('-l', '--len',      default=100, type=int,     help='length of the reference axis')  # FIXME: save and restore from the data
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

if params['cuda']: from phi.tf.tf_cuda_pressuresolver import CUDASolver

from phi.tf.flow import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tf_session = tf.Session(config=config)

from tensorflow import keras

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

random.seed(0 if params['seed'] is None else params['seed'])
np.random.seed(0 if params['seed'] is None else params['seed'])
tf.compat.v1.set_random_seed(0 if params['seed'] is None else params['seed'])

def to_feature(smokestate, ext_const_channel):
    # drop the unused edges of the staggered velocity grid making its dim same to the centered grid's
    with tf.name_scope('to_feature') as scope:
        return math.concat(
            [
                smokestate.velocity.staggered_tensor()[:, :-1:, :-1:, 0:2],
                tf.constant(shape=smokestate.density.data.shape, value=1.0)*math.reshape(value=ext_const_channel, shape=[smokestate._batch_size, 1, 1, 1]),
            ],
            axis=-1
        )

def to_staggered(tensor_cen, box):
    with tf.name_scope('to_staggered') as scope:
        return StaggeredGrid(math.pad(tensor_cen, ((0,0), (0,1), (0,1), (0,0))), box=box)

def model_mercury(tensor_in):
    with tf.name_scope('model_mercury') as scope:
        return keras.Sequential([
            keras.layers.Input(tensor=tensor_in),
            keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu),
            keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation=tf.nn.relu),
            keras.layers.Conv2D(filters=2,  kernel_size=5, padding='same', activation=None),  # u, v
        ])

def model_mars_moon(tensor_in):
    with tf.name_scope('model_mars_moon') as scope:
        l_input = keras.layers.Input(tensor=tensor_in)
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
        return keras.models.Model(inputs=l_input, outputs=l_output)

def downsample4x(tensor):
    return math.downsample2x(math.downsample2x(tensor))

def downsample4xSMAC(tensor):
    return StaggeredGrid(tensor).downsample2x().downsample2x().staggered_tensor()

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


class KarmanFlow(IncompressibleFlow):
    def __init__(self, pressure_solver=None, make_input_divfree=False, make_output_divfree=True):
        IncompressibleFlow.__init__(self, pressure_solver, make_input_divfree, make_output_divfree)

        self.infl = Inflow(box[5:10, 25:75])
        self.obst = Obstacle(Sphere([50, 50], 10))

    def step(self, smoke, re, res, velBCy, velBCyMask, dt=1.0, gravity=Gravity()):
        # apply viscosity
        alpha = 1.0/math.reshape(re, [smoke._batch_size, 1, 1, 1]) * dt * res * res

        cy = diffuse(CenteredGrid(smoke.velocity.data[0].data), alpha)
        cx = diffuse(CenteredGrid(smoke.velocity.data[1].data), alpha)

        # apply velocity BCs, only y for now; velBCy should be pre-multiplied
        cy = cy*(1.0 - velBCyMask) + velBCy

        smoke = smoke.copied_with(velocity=StaggeredGrid([cy.data, cx.data], smoke.velocity.box))

        return super().step(fluid=smoke, dt=dt, obstacles=[self.obst], gravity=gravity, density_effects=[self.infl], velocity_effects=())

class PhifDataset():
    def __init__(self, dirpath, num_frames, num_sims=None, batch_size=1, print_fn=print, skip_preprocessing=False):
        self.dataSims      = sorted(glob.glob(dirpath + '/sim_0*'))[0:num_sims]
        self.pathsDen      = [ sorted(glob.glob(asim + '/dens_0*.npz')) for asim in self.dataSims ]
        self.pathsVel      = [ sorted(glob.glob(asim + '/velo_0*.npz')) for asim in self.dataSims ]
        self.dataFrms      = [ np.arange(num_frames) for _ in self.dataSims ]  # NOTE: may contain different numbers of frames
        self.batchSize     = batch_size
        self.epoch         = None
        self.epochIdx      = 0
        self.batch         = None
        self.batchIdx      = 0
        self.step          = None
        self.stepIdx       = 0
        self.dataPreloaded = None
        self.printFn       = print_fn

        self.numOfSims    = num_sims
        self.numOfBatchs  = self.numOfSims//self.batchSize
        self.numOfFrames  = num_frames
        self.numOfSteps   = num_frames

        if not skip_preprocessing:
            self.printFn('Pre-processing: Loading data from {} = {} and save down-scaled data'.format(dirpath, self.dataSims))
            for j,asim in enumerate(self.dataSims):
                for i in range(num_frames):
                    if not os.path.isfile(self.filenameToDownscaled(self.pathsDen[j][i])):
                        d = downDen(read_zipped_array(self.pathsDen[j][i]))
                        write_zipped_array(self.filenameToDownscaled(self.pathsDen[j][i]), d)
                        self.printFn('Wrote {}'.format(self.filenameToDownscaled(self.pathsDen[j][i])))
                    if not os.path.isfile(self.filenameToDownscaled(self.pathsVel[j][i])):
                        v = downVel(read_zipped_array(self.pathsVel[j][i]))
                        write_zipped_array(self.filenameToDownscaled(self.pathsVel[j][i]), v)
                        self.printFn('Wrote {}'.format(self.filenameToDownscaled(self.pathsVel[j][i])))

        self.printFn('Preload: Loading data from {} = {}'.format(dirpath, self.dataSims))
        self.dataPreloaded = {  # dataPreloaded['sim_key'][step #][0=density, 1=velocity]
            asim: [
                (
                    read_zipped_array(self.filenameToDownscaled(self.pathsDen[j][i])),
                    read_zipped_array(self.filenameToDownscaled(self.pathsVel[j][i])),
                ) for i in range(num_frames)
            ] for j,asim in enumerate(self.dataSims)
        }

        self.resolution = self.dataPreloaded[self.dataSims[0]][0][0].shape[1:3]  # [batch-size, y-size, x-size, dim]
        # TODO: need a sanity check for resolution over all data

        self.dataStats = {
            'std': (
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][0].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)),  # density
                (
                    np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][1][...,0].reshape(-1)) for asim in self.dataSims for i in range(num_frames)])),  # vel[0]
                    np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][1][...,1].reshape(-1)) for asim in self.dataSims for i in range(num_frames)])),  # vel[1]
                )
            )
        }

        self.extConstChannelPerSim = {}  # extConstChannelPerSim['sim_key'][0=first channel, ...]; for now, only Reynolds Nr.
        num_of_ext_channel = 1
        for asim in self.dataSims:
            with open(asim+'/params.pickle', 'rb') as f:
                sim_params = pickle.load(f)
                self.extConstChannelPerSim[asim] = [ sim_params['re'] ]  # Reynolds Nr.

        self.dataStats.update({
            'ext.std': [
                np.std([np.absolute(self.extConstChannelPerSim[asim][i]) for asim in self.dataSims]) for i in range(num_of_ext_channel)  # Reynolds Nr
            ]
        })
        self.printFn(self.dataStats)

    def filenameToDownscaled(self, fname):
        return os.path.dirname(fname) + '/ds_' + os.path.basename(fname)

    def getInstance(self, sim_idx=0, step=0):
        d0_hi = math.concat([self.dataPreloaded[self.dataSims[sim_idx+i]][step][0] for i in range(self.batchSize)], axis=0)
        v0_hi = math.concat([self.dataPreloaded[self.dataSims[sim_idx+i]][step][1] for i in range(self.batchSize)], axis=0)
        # TODO: additional channels
        return [d0_hi, v0_hi]

    def newEpoch(self, exclude_tail=0, shuffle_data=True):
        self.numOfSteps = self.numOfFrames - exclude_tail
        simSteps = [ (asim, self.dataFrms[i][0:(len(self.dataFrms[i])-exclude_tail)]) for i,asim in enumerate(self.dataSims) ]
        sim_step_pair = []
        for i,_ in enumerate(simSteps):
            sim_step_pair += [ (i, astep) for astep in simSteps[i][1] ]  # (sim_idx, step) ...

        if shuffle_data: random.shuffle(sim_step_pair)
        self.epoch = [ list(sim_step_pair[i*self.numOfSteps:(i+1)*self.numOfSteps]) for i in range(self.batchSize*self.numOfBatchs) ]
        self.epochIdx += 1
        self.batchIdx = 0
        self.stepIdx = 0

    def nextBatch(self):  # batch size may be the number of simulations in a batch
        self.batchIdx += self.batchSize
        self.stepIdx = 0

    def nextStep(self):
        self.stepIdx += 1

    def getData(self, consecutive_frames, with_skip=1):
        d_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]  # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip  # steps
                ][0]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        v_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]  # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip  # steps
                ][1]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        ext = [
            self.extConstChannelPerSim[
                self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
            ][0] for i in range(self.batchSize)
        ]
        return [d_hi, v_hi, ext]

    def getPrevData(self, previous_frames, with_skip=1):  # NOTE: not in use; need to test
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
        v_hi = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
                ][
                    max([0, self.epoch[self.batchIdx+i][self.stepIdx][1]-j*with_skip])
                ][1]
                for i in range(self.batchSize)
            ], axis=0) for j in range(previous_frames)
        ]
        # TODO: additional channels
        return [d_hi, v_hi]


downDen = eval('downsample{}x'.format(params['scale']))
downVel = eval('downsample{}xSMAC'.format(params['scale']))
simulator_lo = KarmanFlow()

dataset = PhifDataset(
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

Rlo = list(dataset.resolution)

st_co =   Fluid(Domain(resolution=Rlo, box=box[0:params['len']*2, 0:params['len']], boundaries=OPEN), buoyancy_factor=0, batch_size=params['sbatch'])
st_gt = [ Fluid(Domain(resolution=Rlo, box=box[0:params['len']*2, 0:params['len']], boundaries=OPEN), buoyancy_factor=0, batch_size=params['sbatch']) for _ in range(params['msteps']) ]

# velocity BC
vn = np.zeros(st_co.velocity.data[0].data.shape)  # st.velocity.data[0] is considered as the velocity field in y axis!
vn[..., 0:2, 0:vn.shape[2]-1, 0] = 1.0
vn[..., 0:vn.shape[1], 0:1,   0] = 1.0
vn[..., 0:vn.shape[1], -1:,   0] = 1.0
velBCy = vn
velBCyMask = np.copy(vn)     # warning, only works for 1s, otherwise setup/scale

with tf.name_scope('input') as scope:
    with tf.name_scope('co') as scope: tf_st_co_in =   placeholder(st_co.shape)
    with tf.name_scope('lr') as scope: tf_vr_lr_in =   tf.placeholder(tf.float32, shape=[])  # learning rate
    with tf.name_scope('Re') as scope: tf_st_Re_in =   tf.placeholder(tf.float32, shape=[params['sbatch']])  # Reynolds numbers
    with tf.name_scope('gt') as scope: tf_st_gt_in = [ placeholder(st_co.shape) for _ in range(params['msteps']) ]

if (params['train'] is None):
    log.info(params['train'])
    log.info('No pre-loadable training data path is given.')
    exit(0)

scene = Scene.create(params['train'], count=params['sbatch'], mkdir=False, copy_calling_script=False)
sess = Session(scene, session=tf_session)
tf.compat.v1.keras.backend.set_session(tf_session)

with tf.name_scope('training') as scope:
    netModel = eval('model_{}'.format(params['model']))
    model = netModel(to_feature(tf_st_co_in, tf_st_Re_in))

    with tf.name_scope('corre') as scope:
        tf_st_co_prd, tf_cv_md = [], []
        for i in range(params['msteps']):
            with tf.name_scope('step_w_pred') as scope:
                with tf.name_scope('step') as scope:
                    tf_st_co_prd += [
                        simulator_lo.step(
                            tf_st_co_in if i==0 else tf_st_co_prd[-1],
                            re=tf_st_Re_in,
                            res=Rlo[-1],  # reference resolution is size in x direction
                            velBCy=velBCy,
                            velBCyMask=velBCyMask
                        )
                    ]

                with tf.name_scope('pred') as scope:
                    tf_cv_md += [
                        to_staggered(
                            model(
                                to_feature(tf_st_co_prd[-1], tf_st_Re_in)/[
                                    # 'in.std' and 'out.std' are used in the supervised model, here active only when loading pretf
                                    *(dataset.dataStats['in.std' if 'in.std' in dataset.dataStats else 'std'][1]),  # velocity
                                    dataset.dataStats['ext.std'][0]  # Re
                                ]
                            )*(dataset.dataStats['out.std'] if 'out.std' in dataset.dataStats else dataset.dataStats['std'][1]),
                            box=st_co.velocity.box
                        )
                    ]

                tf_st_co_prd[-1] = tf_st_co_prd[-1].copied_with(velocity=tf_st_co_prd[-1].velocity + tf_cv_md[-1])

    with tf.name_scope('loss') as scope:
        loss_steps = [
            tf.nn.l2_loss(
                (tf_st_gt_in[i].velocity.staggered_tensor() - tf_st_co_prd[i].velocity.staggered_tensor())
                /dataset.dataStats['std'][1]
            )
            for i in range(params['msteps'])
        ]
        loss = tf.reduce_sum(loss_steps)/params['msteps']
        for i,a_step_loss in enumerate(loss_steps): tf.compat.v1.summary.scalar(name='loss_step{:02d}'.format(i), tensor=a_step_loss)
        tf.compat.v1.summary.scalar(name='sum_step_loss', tensor=loss)

        total_loss = loss
        if params['reg_loss']:
            reg_loss = tf.reduce_sum(model.losses)
            total_loss += reg_loss
            tf.compat.v1.summary.scalar(name='regularization', tensor=reg_loss)

        tf.compat.v1.summary.scalar(name='total_loss', tensor=total_loss)
        tf.compat.v1.summary.scalar(name='lr', tensor=tf_vr_lr_in)

        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=tf_vr_lr_in)

        if params['clip_grad']:
            gvs = opt.compute_gradients(total_loss)
            clipped_gvs = [(tf.clip_by_norm(grad, clip_norm=0.001), var) for (grad, var) in gvs]
            train_step = opt.apply_gradients(clipped_gvs)

        else:
            train_step = opt.minimize(total_loss)

model.summary(print_fn=log.info)
sess.initialize_variables()

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

tf_summary_merged = tf.compat.v1.summary.merge_all()
tf_writer_tr = tf.compat.v1.summary.FileWriter(params['tf']+'/summary/training')
if params['resume']<1: tf_writer_tr.add_graph(sess.graph)

current_lr = params['lr']
i_st = 0
for j in range(params['epochs']):  # training
    dataset.newEpoch(exclude_tail=params['msteps'])
    if j<params['resume']:
        log.info('resume: skipping {} epoch'.format(j+1))
        i_st += dataset.numOfSteps*dataset.numOfBatchs
        continue

    current_lr = lr_schedule(j, current_lr) if params['adplr'] else params['lr']
    for ib in range(dataset.numOfBatchs):   # for each batch
        for i in range(dataset.numOfSteps):  # for each step
            adata = dataset.getData(consecutive_frames=params['msteps'], with_skip=1)
            re_nr = adata[2]    # Reynolds numbers
            st_co = st_co.copied_with(density=adata[0][0], velocity=adata[1][0])
            st_gt = [ st_gt[k].copied_with(density=adata[0][k+1], velocity=adata[1][k+1]) for k in range(params['msteps']) ]

            my_feed_dict = { tf_st_co_in: st_co, tf_st_Re_in: re_nr, tf_vr_lr_in: current_lr }
            my_feed_dict.update(zip(tf_st_gt_in, st_gt))
            summary, _, l2 = sess.run([tf_summary_merged, train_step, total_loss], my_feed_dict)

            tf_writer_tr.add_summary(summary, i_st)
            i_st += 1

            log.info('epoch {:03d}/{:03d}, batch {:03d}/{:03d}, step {:04d}/{:04d}: loss={}'.format(
                j+1, params['epochs'], ib+1, dataset.numOfBatchs, i+1, dataset.numOfSteps, l2
            ))
            dataset.nextStep()

        dataset.nextBatch()

    if j%10==9: model.save(params['tf']+'/model_epoch{:04d}.h5'.format(j+1))

tf_writer_tr.close()
model.save(params['tf']+'/model.h5')
