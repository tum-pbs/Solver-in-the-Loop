# ----------------------------------------------------------------------------
#
# Phiflow Karman vortex solver framework
# Copyright 2019-2020 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Training (PRE versions)
#
# ----------------------------------------------------------------------------

import os, sys, glob, pickle, argparse, logging

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

import numpy as np

# default parameters
params = {}
parser = argparse.ArgumentParser(description='Parse parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--opath',                   default='/tmp/tf_test_model', help='output path')
parser.add_argument('--tmp',                           default=None,                 help='temporary directory; will override the system tmp')
parser.add_argument('--nolog',                         action='store_true',          help='turn off logging')
parser.add_argument('--nopng',                         action='store_true',          help='turn off saving PNGs')
parser.add_argument('--nostats',                       action='store_true',          help='turn off stats')
parser.add_argument('--notrain',                       action='store_true',          help='turn off training')
parser.add_argument('--novdata',                       action='store_true',          help='turn off use of validation data')
parser.add_argument('--nogpu',                         action='store_true',          help='turn off GPU')
parser.add_argument('--nozerocen',                     action='store_true',          help='normalize data without zero-centered')
parser.add_argument('--augment',                       action='store_true',          help='apply data augmentation')
parser.add_argument('--nsigma',                        default=1.0,  type=float,     help='normalize (or standardize) data with this sigma value')
parser.add_argument('--val',                           default=0.2,  type=float,     help='validation data (split) size')
parser.add_argument('--bsize', dest='batch_size',      default=32,   type=int,       help='batch size')
parser.add_argument('--epochs',                        default=1000, type=int,       help='number of epoch')
parser.add_argument('--seed',                          default=None, type=int,       help='seed for random number generator')
parser.add_argument('--steps', dest='steps_per_epoch', default=None, type=int,       help='how many steps (i.e., batches) per epoch')
parser.add_argument('--lr',                            default=1e-3, type=float,     help='start learning rate')
parser.add_argument('--model',                         default='mars_moon',          help='model name')
parser.add_argument('--inftr',                         default='scandium',           help='input feature')
parser.add_argument('-k', '--keep',                    default=None,                 help='keep old model if exists')
parser.add_argument('tdata',                           action='store', nargs='+',    help='paths containing npz files for training data')
pargs = parser.parse_args()
params.update(vars(pargs))

if params['nogpu']: os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow import keras

if not params['nogpu']:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU; to solve the problem of "Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED"
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

np.random.seed(0 if params['seed'] is None else params['seed'])
tf.compat.v1.set_random_seed(0 if params['seed'] is None else params['seed'])

paths        = {}
paths['mdl'] = params['opath'] + '/model.h5'
paths['mdp'] = params['opath'] + '/model.png'
paths['mck'] = params['opath'] + '/model.ckh5'
paths['tsb'] = params['opath'] + '/logs'
paths['log'] = params['opath'] + '/run.log'
paths['png'] = params['opath'] + '/stats-png'
paths['pdf'] = params['opath'] + '/stats.pdf'
paths['pkl'] = params['opath'] + '/stats.pickle'

if params['keep'] is not None:
    os.path.exists(params['opath']) and os.rename(params['opath'], params['keep']+'-'+params['opath'])
else:
    import shutil, tempfile, datetime
    tkey = datetime.datetime.now().strftime('%Y%m%d%H%M')
    tmp = tempfile.gettempdir() if pargs.tmp is None else pargs.tmp
    os.path.exists(params['opath']) and shutil.move(params['opath'], tmp+'/{}-{}'.format(tkey, params['opath'].replace('/', '-')))

os.path.exists(params['opath']) or os.makedirs(params['opath'])

##############################################################################
# data

def read_grid(path, rtbnd=-1, dtype='float32'):
    # rtbnd: manually correct a potential shape mismatch
    if path.endswith('.npz'):
        npdata = np.load(path)['arr_0']
        head = {
            'dimX': npdata.shape[-2] + rtbnd,
            'dimY': npdata.shape[-3] + rtbnd,
            'dimZ': 1,
        }
        if npdata.shape[-1]>1:
            npdata = np.concatenate(
                (npdata[:-1,:-1,:].reshape((head['dimZ'], head['dimY'], head['dimX'], npdata.shape[-1])),
                 np.zeros(shape=(head['dimZ'], head['dimY'], head['dimX'], 1))),
                axis=-1
            )

        else:
            npdata = npdata.reshape((head['dimZ'], head['dimY'], head['dimX']))

        return head, npdata

    else:
        print('Filetype is not supported.')
        exit(0)


from scipy import stats
import tf_data as dmani

files = {
    # inputs
    'fld': [ f for ffs in (sorted(glob.glob('{}/dens_0*.npz'.format(i))) for i in params['tdata']) for f in ffs ],
    'vel': [ f for ffs in (sorted(glob.glob('{}/velo_0*.npz'.format(i))) for i in params['tdata']) for f in ffs ],

    # outputs
    'Vco': [ f for ffs in (sorted(glob.glob('{}/corr_0*.npz'.format(i))) for i in params['tdata']) for f in ffs ],
}

assert all([ (len(files[i])==len(files['Vco'])) for i in files ]), 'Some data files are missing'

data_stats = {}

def lr_schedule(epoch, current_lr):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = current_lr
    if   epoch == 181: lr *= 0.5
    elif epoch == 161: lr *= 1e-1
    elif epoch == 121: lr *= 1e-1
    elif epoch == 81:  lr *= 1e-1
    return lr

def getPrevFile(curr_i, offset, filekey, files):
    prev_i = max([0, curr_i + offset])
    if os.path.dirname(files[filekey][prev_i])==os.path.dirname(files[filekey][curr_i]):
        return files[filekey][prev_i]

    else:
        for j in range(offset, 1, 1):
            if os.path.dirname(files[filekey][curr_i+j])==os.path.dirname(files[filekey][curr_i]):
                return files[filekey][curr_i+j]

    return files[filekey][curr_i]

def extractRe(curr_i):
    with open(os.path.dirname(files['Vco'][curr_i])+'/params.pickle', 'rb') as f: params = pickle.load(f)
    return params['re']

def data_scandium():
    data_stats['nG'] = 1
    data_stats['augment_flipX'] = [1.0,-1.0, 1.0, 1.0,-1.0]  # v(n),u(n), Re, vcorr_v,vcorr_u
    head, _ = read_grid(files['Vco'][0], -1, 'float32')
    def ainput(i):
        print('{}/{}'.format(i+1, len(files['vel'])), end='\r', flush=True)
        return np.concatenate(
            (
             read_grid(files['vel'][i], -1, 'float32')[1][...,0:2][...,::-1].reshape((1, head['dimY'], head['dimX'], 2)),  # revert indices! for phiflow
             np.ones(shape=(1, head['dimY'], head['dimX'], 1))*extractRe(i)
            ),
            axis=3
        )
    inputs = [ ainput(i) for i,_ in enumerate(files['Vco']) ]
    labels = [ read_grid(afile, -1, 'float32')[1][...,0:2][...,::-1].reshape((1, head['dimY'], head['dimX'], 2)) for afile in files['Vco'] ]  # revert indices! for phiflow
    data = {
        'inputs': np.concatenate(tuple(inputs), axis=0),
        'labels': np.concatenate(tuple(labels), axis=0),
    }
    return head, data

def augment_random_flip_sample(features, label):
    asample = tf.concat([features, label], axis=-1)
    do_flip = tf.random.uniform([]) > 0.5
    nsample = tf.cond(do_flip, lambda: tf.image.flip_left_right(asample)*data_stats['augment_flipX'], lambda: asample)
    return nsample[..., 0:-2], nsample[..., -2:]  # input, label


print('\nLoading data ...', flush=True)
head, data  = eval('data_{}()'.format(params['inftr']))
nX, nY, nF  = head['dimX'], head['dimY'], data['inputs'].shape[-1]
data_stats.update({ 'feature': params['inftr'], 'nX': nX, 'nY': nY, 'nF': nF })
print('\n... Done', flush=True)

[ params['nolog'] or log.addHandler(logging.FileHandler(paths['log'])) ]
log.info(params)
log.info(paths)
log.info('{} -> {}'.format(data['inputs'].shape, data['labels'].shape))

perm = np.arange(data['labels'].shape[0])
np.random.shuffle(perm)

val_size = int(params['val']*data['labels'].shape[0]) if not params['novdata'] else data['labels'].shape[0]
train_inputs = data['inputs'][perm][:-val_size] if not params['novdata'] else data['inputs'][perm]
train_labels = data['labels'][perm][:-val_size] if not params['novdata'] else data['labels'][perm]
valid_inputs = data['inputs'][perm][-val_size:] if not params['novdata'] else data['inputs'][perm]
valid_labels = data['labels'][perm][-val_size:] if not params['novdata'] else data['labels'][perm]

log.info('training: {} -> {}'.format(train_inputs.shape, train_labels.shape))
log.info('validation: {} -> {}'.format(valid_inputs.shape, valid_labels.shape))

print('\nNormalizing data ...', flush=True)
data_stats.update(dmani.dataStats(idata=train_inputs, odata=train_labels))
data_stats.update({'nozerocen': params['nozerocen'], 'nsigma': params['nsigma']})
log.info(data_stats)
if not params['nostats']:
    with open(paths['pkl'], 'wb') as f: pickle.dump(data_stats, f)

dmani.standardize(idata=train_inputs, odata=train_labels, dstats=data_stats, sigma_range=params['nsigma'], zero_centered=(not params['nozerocen']))
dmani.standardize(idata=valid_inputs, odata=valid_labels, dstats=data_stats, sigma_range=params['nsigma'], zero_centered=(not params['nozerocen']))

log.info(stats.describe(train_inputs, axis=None))
log.info(stats.describe(train_labels, axis=None))
log.info(stats.describe(valid_inputs, axis=None))
log.info(stats.describe(valid_labels, axis=None))
print('... Done', flush=True)

# data statistics
if not params['nostats']:
    print('\nPlotting data statistics ...', flush=True, end='')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.backends.backend_pdf import PdfPages

    params['nopng'] or os.path.exists(paths['png']) or os.makedirs(paths['png'])
    with PdfPages(paths['pdf']) as pdf:
        for i in data:
            for j in range(data[i].shape[-1]):
                dd = data[i][...,j].reshape(-1)
                ss = stats.describe(dd)
                plt.figure()
                plt.hist(dd, bins=100, log=True)
                plt.gca().text(0.5, 1.0, ss, wrap=True, horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes, fontsize=8)
                plt.title('Histogram of {}[{}]'.format(i, j))
                plt.savefig(pdf, format='pdf')
                params['nopng'] or plt.savefig('{}/{}_{}.png'.format(os.path.normpath(paths['png']), i, j))
                plt.close('all')

        for j in range(train_inputs.shape[-1]):
            dd = train_inputs[...,j].reshape(-1)
            ss = stats.describe(dd)
            plt.figure()
            plt.hist(dd, bins=100, log=True)
            plt.gca().text(0.5, 1.0, ss, wrap=True, horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes, fontsize=8)
            plt.title('Histogram of input_train[{}] (normalized)'.format(j))
            plt.savefig(pdf, format='pdf')
            params['nopng'] or plt.savefig('{}/input_train_{}_normalized.png'.format(os.path.normpath(paths['png']), j))
            plt.close('all')

        for j in range(train_labels.shape[-1]):
            dd = train_labels[...,j].reshape(-1)
            ss = stats.describe(dd)
            plt.figure()
            plt.hist(dd, bins=100, log=True)
            plt.gca().text(0.5, 1.0, ss, wrap=True, horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes, fontsize=8)
            plt.title('Histogram of label_train[{}] (normalized)'.format(j))
            plt.savefig(pdf, format='pdf')
            params['nopng'] or plt.savefig('{}/label_train_{}_normalized.png'.format(os.path.normpath(paths['png']), j))
            plt.close('all')

    print(' Done', flush=True)

def model_mercury():
    return keras.Sequential([
        keras.layers.Input(shape=(nY, nX, nF)),
        keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu),
        keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation=tf.nn.relu),
        keras.layers.Conv2D(filters=2,  kernel_size=5, padding='same', activation=None),  # u, v
    ])

def model_mars_moon():
    l_input = keras.layers.Input(shape=(nY, nX, nF))
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


from tensorflow.python.eager import context
class TrainValTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

    def set_model(self, model):
        if context.executing_eagerly():
            self.val_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)
        else:
            self.val_writer = tf.compat.v1.summary.FileWriter(self.val_log_dir)

        super(TrainValTensorBoard, self).set_model(model)

    def _write_custom_summaries(self, step, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
        if context.executing_eagerly():
            with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in val_logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)
        else:
            for name, value in val_logs.items():
                summary = tf.compat.v1.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, step)

        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not 'val_' in k}
        super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

class XTensorBoard(TrainValTensorBoard):
    def lr_getter(self):
        return np.float32(keras.backend.eval(self.model.optimizer.lr))

    def on_epoch_end(self, episode, logs = {}):
        logs.update({"lr": self.lr_getter()})
        super(XTensorBoard, self).on_epoch_end(episode, logs)


if not params['notrain']:
    log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

    dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    if params['augment']: dataset = dataset.map(augment_random_flip_sample)
    dataset = dataset.shuffle(buffer_size=train_labels.shape[0]).batch(params['batch_size']).repeat()
    valset  = tf.data.Dataset.from_tensor_slices((valid_inputs, valid_labels)).batch(round(val_size*0.1)).repeat()

    model = eval('model_{}()'.format(params['model']))

    opt = keras.optimizers.Adam(lr=lr_schedule(epoch=0, current_lr=params['lr']))
    model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
    model.summary(print_fn=log.info)
    keras.utils.plot_model(model, to_file=paths['mdp'], show_shapes=True)

    mycallbacks = [
        XTensorBoard(log_dir=paths['tsb'], histogram_freq=10, batch_size=params['batch_size']),
        keras.callbacks.ModelCheckpoint(paths['mck'], monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1),
        keras.callbacks.LearningRateScheduler(lr_schedule),
    ]

    model.fit(
        dataset,
        epochs=params['epochs'],
        steps_per_epoch=params['steps_per_epoch'] if params['steps_per_epoch'] else round(train_labels.shape[0]/params['batch_size']),
        validation_data=valset, validation_steps=10,
        callbacks=mycallbacks)

    log.info(model.evaluate(valid_inputs, valid_labels))
    log.info(stats.describe(model.predict(valid_inputs), axis=None))

    model.save(paths['mdl'])

lmodel = keras.models.load_model(paths['mdl'])
lmodel.summary(print_fn=log.info)
log.info(stats.describe(lmodel.predict(valid_inputs), axis=None))
