# ----------------------------------------------------------------------------
#
# Phiflow Karman vortex solver framework
# Copyright 2020 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Data generation (PRE versions)
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
parser.add_argument('-o', '--output',    default=None,            help='path to an output directory')
parser.add_argument('--thumb',           action='store_true',     help='save thumbnail images')
parser.add_argument('-t', '--simsteps',  default=1500, type=int,  help='simulation steps: an epoch')
parser.add_argument('-s', '--skipsteps', default=999, type=int,   help='skip first steps; (vortices may not form)')
parser.add_argument('-r', '--res',       default=32, type=int,    help='resolution of the reference axis')
parser.add_argument('-l', '--len',       default=100, type=int,   help='length of the reference axis')
parser.add_argument('--scale',           default=4, type=int,     help='simulation scale for high-res')
parser.add_argument('--re',              default=1e6, type=float, help='Reynolds number')
parser.add_argument('--seed',            default=0, type=int,     help='seed for random number generator')
parser.add_argument('--beta',            default=1.0, type=float, help='temporal regularizer')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

from phi.flow import *

random.seed(params['seed'])
np.random.seed(params['seed'])

import scipy.sparse.linalg
from scipy import interpolate

def downsample4x(tensor):
    return math.downsample2x(math.downsample2x(tensor))

def upsample4x(tensor):
    return math.upsample2x(math.upsample2x(tensor))

def downsample4xSMAC(tensor):
    return StaggeredGrid(tensor).downsample2x().downsample2x().staggered_tensor()

def upsample2xSMAC(tensor):     # NOTE: assumed the batch size is 1!!!
    tshape = (tensor.shape[0], (tensor.shape[-3]-1)*2+1, (tensor.shape[-2]-1)*2+1, 2)
    lo_x, lo_y = np.arange(0.0, tensor.shape[2]), np.arange(0.0, tensor.shape[1])
    hi_x, hi_y = np.arange(0.0, tshape[2])/2, np.arange(0.0, tshape[1])/2

    up_u = interpolate.interp2d(lo_x, lo_y+0.5, tensor[0, :, :, 1], kind='linear')(hi_x, hi_y+(0.5/2))
    up_v = interpolate.interp2d(lo_x+0.5, lo_y, tensor[0, :, :, 0], kind='linear')(hi_x+(0.5/2), hi_y)

    new_tensor = math.concat(
        [math.reshape(up_v, (tshape[0], tshape[-3], tshape[-2], 1)),
         math.reshape(up_u, (tshape[0], tshape[-3], tshape[-2], 1))], axis=-1  # phiflow stack v first and u second
    )

    return new_tensor

def upsample4xSMAC(tensor):
    return upsample2xSMAC(upsample2xSMAC(tensor))


downDen = eval('downsample{}x'.format(params['scale']))
downVel = eval('downsample{}xSMAC'.format(params['scale']))
upDen = eval('upsample{}x'.format(params['scale']))
upVel = eval('upsample{}xSMAC'.format(params['scale']))

def fluidCellIndexes(tensor_cen, bnd):
    # just fill all cells except boundary (bnd) for now, may consider density values afterwards?
    cnt, npg = 0, np.ones(shape=(1, tensor_cen.shape[-3], tensor_cen.shape[-2]), dtype=np.int32)*-1

    for j in np.arange(bnd, tensor_cen.shape[-3]-bnd):
        for i in np.arange(bnd, tensor_cen.shape[-2]-bnd):
            npg[0, j, i] = cnt
            cnt += 1

    return cnt, npg

def fluidFaceIndexes(cen_index, dim, bnd=1):
    # if at least one adjacent cell has a valid index
    cnt, npg = [0]*dim, [np.ones(shape=cen_index.shape, dtype=np.int32)*-1 for _ in range(dim)]

    for k in (np.arange(bnd, cen_index.shape[-3]-bnd) if dim>2 else np.arange(1)):
        for j in np.arange(bnd, cen_index.shape[-2]-bnd):
            for i in np.arange(bnd, cen_index.shape[-1]-bnd):
                if (cen_index[k, j, i]>-1) or (cen_index[k, j, i-1]>-1):
                    npg[0][k, j, i] = cnt[0]
                    cnt[0] += 1

                if (cen_index[k, j, i]>-1) or (cen_index[k, j-1, i]>-1):
                    npg[1][k, j, i] = cnt[1]
                    cnt[1] += 1

                if dim<3: continue
                if (cen_index[k, j, i]>-1) or (cen_index[k-1, j, i]>-1):
                    npg[2][k, j, i] = cnt[2]
                    cnt[2] += 1

    return cnt, npg

def fillMatW(paramlist):
    ii, jj, kk, cnt_l, npg_l, cnt_h, npg_h, npg_vh, dim = paramlist
    w_row, w_col, w_data, mat_vh = [], [], [], []
    DD = pow(2, dim)
    sfH = [params['scale'], params['scale'], params['scale'] if dim>2 else 1]
    if npg_h[0][kk, jj, ii]>=0:  # u-component
        x, y, z = ii/sfH[0], (jj+0.5)/sfH[1], (kk+0.5)/sfH[2]
        i, j, k = int(x), int(y), int(z)
        fx, fy, fz  = x - i,  y - j,  z - k

        ih,  jh,  kh  = int(x-0.5), int(y-0.5), int(z-0.5)
        fxh, fyh, fzh = x-0.5 - ih, y-0.5 - jh, z-0.5 - kh

        w_row += [(npg_h[0][kk, jj, ii]*DD+0, npg_h[0][kk, jj, ii])]
        w_row += [(npg_h[0][kk, jj, ii]*DD+1, npg_h[0][kk, jj, ii])]
        w_row += [(npg_h[0][kk, jj, ii]*DD+2, npg_h[0][kk, jj, ii])]
        w_row += [(npg_h[0][kk, jj, ii]*DD+3, npg_h[0][kk, jj, ii])]
        if dim>2:
            w_row += [(npg_h[0][kk, jj, ii]*DD+4, npg_h[0][kk, jj, ii])]
            w_row += [(npg_h[0][kk, jj, ii]*DD+5, npg_h[0][kk, jj, ii])]
            w_row += [(npg_h[0][kk, jj, ii]*DD+6, npg_h[0][kk, jj, ii])]
            w_row += [(npg_h[0][kk, jj, ii]*DD+7, npg_h[0][kk, jj, ii])]

        w, c = 0, 0
        if npg_l[0][kh, jh,   i  ]>-1: w_col += [(npg_h[0][kk, jj, ii]*DD+0, npg_l[0][kh, jh,   i  ])]; w_data += [[npg_h[0][kk, jj, ii]*DD+0, (1.0 - fx)*(1.0 - fyh)*(1.0 - fzh)]]; w += w_data[-1][1]; c += 1
        if npg_l[0][kh, jh,   i+1]>-1: w_col += [(npg_h[0][kk, jj, ii]*DD+1, npg_l[0][kh, jh,   i+1])]; w_data += [[npg_h[0][kk, jj, ii]*DD+1,        fx *(1.0 - fyh)*(1.0 - fzh)]]; w += w_data[-1][1]; c += 1
        if npg_l[0][kh, jh+1, i  ]>-1: w_col += [(npg_h[0][kk, jj, ii]*DD+2, npg_l[0][kh, jh+1, i  ])]; w_data += [[npg_h[0][kk, jj, ii]*DD+2, (1.0 - fx)*       fyh *(1.0 - fzh)]]; w += w_data[-1][1]; c += 1
        if npg_l[0][kh, jh+1, i+1]>-1: w_col += [(npg_h[0][kk, jj, ii]*DD+3, npg_l[0][kh, jh+1, i+1])]; w_data += [[npg_h[0][kk, jj, ii]*DD+3,        fx *       fyh *(1.0 - fzh)]]; w += w_data[-1][1]; c += 1
        if dim>2:
            if npg_l[0][kh+1, jh,   i  ]>-1: w_col += [(npg_h[0][kk, jj, ii]*DD+4, npg_l[0][kh+1, jh,   i  ])]; w_data += [[npg_h[0][kk, jj, ii]*DD+4, (1.0 - fx)*(1.0 - fyh)*fzh]]; w += w_data[-1][1]; c += 1
            if npg_l[0][kh+1, jh,   i+1]>-1: w_col += [(npg_h[0][kk, jj, ii]*DD+5, npg_l[0][kh+1, jh,   i+1])]; w_data += [[npg_h[0][kk, jj, ii]*DD+5,        fx *(1.0 - fyh)*fzh]]; w += w_data[-1][1]; c += 1
            if npg_l[0][kh+1, jh+1, i  ]>-1: w_col += [(npg_h[0][kk, jj, ii]*DD+6, npg_l[0][kh+1, jh+1, i  ])]; w_data += [[npg_h[0][kk, jj, ii]*DD+6, (1.0 - fx)*       fyh *fzh]]; w += w_data[-1][1]; c += 1
            if npg_l[0][kh+1, jh+1, i+1]>-1: w_col += [(npg_h[0][kk, jj, ii]*DD+7, npg_l[0][kh+1, jh+1, i+1])]; w_data += [[npg_h[0][kk, jj, ii]*DD+7,        fx *       fyh *fzh]]; w += w_data[-1][1]; c += 1

        for cc in range(c): w_data[-cc-1][1] /= w

        mat_vh += [(npg_h[0][kk, jj, ii], npg_vh[kk, jj, ii, 0])]

    if npg_h[1][kk, jj, ii]>=0:  # v-component
        x, y, z = (ii+0.5)/sfH[0], jj/sfH[1], (kk+0.5)/sfH[2]
        i, j, k = int(x), int(y), int(z)
        fx, fy, fz  = x - i,  y - j,  z - k

        ih,  jh,  kh  = int(x-0.5), int(y-0.5), int(z-0.5)
        fxh, fyh, fzh = x-0.5 - ih, y-0.5 - jh, z-0.5 - kh

        hidx_off, lidx_off = cnt_h[0], cnt_l[0]
        w_row += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+0, npg_h[1][kk, jj, ii]+hidx_off)]
        w_row += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+1, npg_h[1][kk, jj, ii]+hidx_off)]
        w_row += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+2, npg_h[1][kk, jj, ii]+hidx_off)]
        w_row += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+3, npg_h[1][kk, jj, ii]+hidx_off)]
        if dim>2:
            w_row += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+4, npg_h[1][kk, jj, ii]+hidx_off)]
            w_row += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+5, npg_h[1][kk, jj, ii]+hidx_off)]
            w_row += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+6, npg_h[1][kk, jj, ii]+hidx_off)]
            w_row += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+7, npg_h[1][kk, jj, ii]+hidx_off)]

        w, c = 0, 0
        if npg_l[1][kh, j,   ih  ]>-1: w_col += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+0, npg_l[1][kh, j,   ih  ]+lidx_off)]; w_data += [[(npg_h[1][kk, jj, ii]+hidx_off)*DD+0, (1.0 - fxh)*(1.0 - fy)*(1.0 - fzh)]]; w += w_data[-1][1]; c += 1
        if npg_l[1][kh, j,   ih+1]>-1: w_col += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+1, npg_l[1][kh, j,   ih+1]+lidx_off)]; w_data += [[(npg_h[1][kk, jj, ii]+hidx_off)*DD+1,        fxh *(1.0 - fy)*(1.0 - fzh)]]; w += w_data[-1][1]; c += 1
        if npg_l[1][kh, j+1, ih  ]>-1: w_col += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+2, npg_l[1][kh, j+1, ih  ]+lidx_off)]; w_data += [[(npg_h[1][kk, jj, ii]+hidx_off)*DD+2, (1.0 - fxh)*       fy *(1.0 - fzh)]]; w += w_data[-1][1]; c += 1
        if npg_l[1][kh, j+1, ih+1]>-1: w_col += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+3, npg_l[1][kh, j+1, ih+1]+lidx_off)]; w_data += [[(npg_h[1][kk, jj, ii]+hidx_off)*DD+3,        fxh *       fy *(1.0 - fzh)]]; w += w_data[-1][1]; c += 1
        if dim>2:
            if npg_l[1][kh+1, j,   ih  ]>-1: w_col += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+4, npg_l[1][kh+1, j,   ih  ]+lidx_off)]; w_data += [[(npg_h[1][kk, jj, ii]+hidx_off)*DD+4, (1.0 - fxh)*(1.0 - fy)*fzh]]; w += w_data[-1][1]; c += 1
            if npg_l[1][kh+1, j,   ih+1]>-1: w_col += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+5, npg_l[1][kh+1, j,   ih+1]+lidx_off)]; w_data += [[(npg_h[1][kk, jj, ii]+hidx_off)*DD+5,        fxh *(1.0 - fy)*fzh]]; w += w_data[-1][1]; c += 1
            if npg_l[1][kh+1, j+1, ih  ]>-1: w_col += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+6, npg_l[1][kh+1, j+1, ih  ]+lidx_off)]; w_data += [[(npg_h[1][kk, jj, ii]+hidx_off)*DD+6, (1.0 - fxh)*       fy *fzh]]; w += w_data[-1][1]; c += 1
            if npg_l[1][kh+1, j+1, ih+1]>-1: w_col += [((npg_h[1][kk, jj, ii]+hidx_off)*DD+7, npg_l[1][kh+1, j+1, ih+1]+lidx_off)]; w_data += [[(npg_h[1][kk, jj, ii]+hidx_off)*DD+7,        fxh *       fy *fzh]]; w += w_data[-1][1]; c += 1

        for cc in range(c): w_data[-cc-1][1] /= w

        mat_vh += [(npg_h[1][kk, jj, ii]+hidx_off, npg_vh[kk, jj, ii, 1])]

    if dim>2 and npg_h[2][kk, jj, ii]>=0:  # w-component
        x, y, z = (ii+0.5)/sfH[0], (jj+0.5)/sfH[1], kk/sfH[2]
        i, j, k = int(x), int(y), int(z)
        fx, fy, fz  = x - i,  y - j,  z - k

        ih,  jh,  kh  = int(x-0.5), int(y-0.5), int(z-0.5)
        fxh, fyh, fzh = x-0.5 - ih, y-0.5 - jh, z-0.5 - kh

        hidx_off, lidx_off = cnt_h[0]+cnt_h[1], cnt_l[0]+cnt_l[1]
        w_row += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+0, npg_h[2][kk, jj, ii]+hidx_off)]
        w_row += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+1, npg_h[2][kk, jj, ii]+hidx_off)]
        w_row += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+2, npg_h[2][kk, jj, ii]+hidx_off)]
        w_row += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+3, npg_h[2][kk, jj, ii]+hidx_off)]
        w_row += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+4, npg_h[2][kk, jj, ii]+hidx_off)]
        w_row += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+5, npg_h[2][kk, jj, ii]+hidx_off)]
        w_row += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+6, npg_h[2][kk, jj, ii]+hidx_off)]
        w_row += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+7, npg_h[2][kk, jj, ii]+hidx_off)]

        w, c = 0, 0
        if npg_l[2][k,   jh,   ih  ]>-1: w_col += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+0, npg_l[2][kh,   j,   ih  ]+lidx_off)]; w_data += [[(npg_h[2][kk, jj, ii]+hidx_off)*DD+0, (1.0 - fxh)*(1.0 - fyh)*(1.0 - fz)]]; w += w_data[-1][1]; c += 1
        if npg_l[2][k,   jh,   ih+1]>-1: w_col += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+1, npg_l[2][kh,   j,   ih+1]+lidx_off)]; w_data += [[(npg_h[2][kk, jj, ii]+hidx_off)*DD+1,        fxh *(1.0 - fyh)*(1.0 - fz)]]; w += w_data[-1][1]; c += 1
        if npg_l[2][k,   jh+1, ih  ]>-1: w_col += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+2, npg_l[2][kh,   j+1, ih  ]+lidx_off)]; w_data += [[(npg_h[2][kk, jj, ii]+hidx_off)*DD+2, (1.0 - fxh)*       fyh *(1.0 - fz)]]; w += w_data[-1][1]; c += 1
        if npg_l[2][k,   jh+1, ih+1]>-1: w_col += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+3, npg_l[2][kh,   j+1, ih+1]+lidx_off)]; w_data += [[(npg_h[2][kk, jj, ii]+hidx_off)*DD+3,        fxh *       fyh *(1.0 - fz)]]; w += w_data[-1][1]; c += 1
        if npg_l[2][k+1, jh,   ih  ]>-1: w_col += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+4, npg_l[2][kh+1, j,   ih  ]+lidx_off)]; w_data += [[(npg_h[2][kk, jj, ii]+hidx_off)*DD+4, (1.0 - fxh)*(1.0 - fyh)*       fz ]]; w += w_data[-1][1]; c += 1
        if npg_l[2][k+1, jh,   ih+1]>-1: w_col += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+5, npg_l[2][kh+1, j,   ih+1]+lidx_off)]; w_data += [[(npg_h[2][kk, jj, ii]+hidx_off)*DD+5,        fxh *(1.0 - fyh)*       fz ]]; w += w_data[-1][1]; c += 1
        if npg_l[2][k+1, jh+1, ih  ]>-1: w_col += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+6, npg_l[2][kh+1, j+1, ih  ]+lidx_off)]; w_data += [[(npg_h[2][kk, jj, ii]+hidx_off)*DD+6, (1.0 - fxh)*       fyh *       fz ]]; w += w_data[-1][1]; c += 1
        if npg_l[2][k+1, jh+1, ih+1]>-1: w_col += [((npg_h[2][kk, jj, ii]+hidx_off)*DD+7, npg_l[2][kh+1, j+1, ih+1]+lidx_off)]; w_data += [[(npg_h[2][kk, jj, ii]+hidx_off)*DD+7,        fxh *       fyh *       fz ]]; w += w_data[-1][1]; c += 1
        for cc in range(c): w_data[-cc-1][1] /= w

        mat_vh += [(npg_h[2][kk, jj, ii]+hidx_off, npg_vh[kk, jj, ii, 2])]

    return w_row, w_col, w_data, mat_vh

import itertools, multiprocessing
def solveVCorrLMopt(corr_prev, dens_lo, dens_hi, velo_hi, beta=0, debug=0):
    # all grid data should be given as phiflow tensors, i.e., (batch_size, size_y, size_x, values_dim)
    # be careful with the indexes of phiflow and mantaflow (reused here); 3D doesn't support for now!
    gs    = [dens_lo.shape[-2], dens_lo.shape[-3], 1]  # x, y, and z
    gs_hi = [dens_hi.shape[-2], dens_hi.shape[-3], 1]  # x, y, and z
    dim   = velo_hi.shape[-1]

    Ng,  npgC  = fluidCellIndexes(tensor_cen=dens_lo, bnd=2)  # mark fluid cells (lores)
    NgH, npgCH = fluidCellIndexes(tensor_cen=dens_hi, bnd=2*params['scale'])  # mark fluid cells (hires); instead of old codes' magnification

    cntL, npgL = fluidFaceIndexes(cen_index=npgC,  dim=dim)
    cntH, npgH = fluidFaceIndexes(cen_index=npgCH, dim=dim)
    Nrow = sum(cntH)          # interpolations: # of hires valid face velocities
    Ncol = sum(cntL)          # # of lores valid face velocities

    vh =  np.concatenate(       # use like a mantaflow SMAC grid
        (velo_hi[0, :-1, :-1, 1:2].reshape((gs_hi[2], gs_hi[1], gs_hi[0], 1)),  # tensor[..., 1]: u
         velo_hi[0, :-1, :-1, 0:1].reshape((gs_hi[2], gs_hi[1], gs_hi[0], 1)),  # tensor[..., 0]: v
         np.zeros(shape=(gs_hi[2], gs_hi[1], gs_hi[0], 1))), axis=-1  # zeros for w
    )

    vl_prev = np.concatenate(
        (corr_prev[0, :-1, :-1, 1:2].reshape((gs[2], gs[1], gs[0], 1)),  # tensor[..., 1]: u
         corr_prev[0, :-1, :-1, 0:1].reshape((gs[2], gs[1], gs[0], 1)),  # tensor[..., 0]: v
         np.zeros(shape=(gs[2], gs[1], gs[0], 1))), axis=-1  # zeros for w
    )

    print('Build W matrix: ', end='', flush=True)
    rowW = np.zeros(shape=Nrow*pow(2, dim), dtype=np.int32)
    colW = np.zeros(shape=Nrow*pow(2, dim), dtype=np.int32)
    dataW = np.zeros(shape=Nrow*pow(2, dim), dtype=np.float32)
    matVh = np.zeros(shape=(Nrow, 1), dtype=np.float32)
    paramlist = list(itertools.product(range(gs_hi[0]), range(gs_hi[1]), range(gs_hi[2]), [cntL], [npgL], [cntH], [npgH], [vh], [dim]))
    with multiprocessing.Pool(multiprocessing.cpu_count()+1) as p:
        returns = p.map(fillMatW, paramlist)
        for w_row, w_col, w_data, mat_vh in returns:
            for ains in w_row:  rowW[ains[0]] = ains[1]
            for ains in w_col:  colW[ains[0]] = ains[1]
            for ains in w_data: dataW[ains[0]] = ains[1]
            for ains in mat_vh: matVh[ains[0]] = ains[1]

    matW = scipy.sparse.csr_matrix((dataW, (rowW, colW)), shape=(Nrow, Ncol), dtype=np.float32)
    print('Done', flush=True)

    print('Build Grad matrix: ', end='', flush=True)
    mat2Vl = np.zeros(shape=(Ncol, 1), dtype=np.float32)
    rowG = np.zeros(shape=Ncol*2, dtype=np.int32)
    colG = np.zeros(shape=Ncol*2, dtype=np.int32)
    dataG = np.zeros(shape=Ncol*2, dtype=np.float32)
    for kk in range(gs[2]):
        for jj in range(gs[1]):
            for ii in range(gs[0]):
                if (npgL[0][kk, jj, ii]>-1):
                    mat2Vl[npgL[0][kk, jj, ii]] = vl_prev[kk, jj, ii, 0]*2
                    rowG[npgL[0][kk, jj, ii]*2+0] = npgL[0][kk, jj, ii]
                    rowG[npgL[0][kk, jj, ii]*2+1] = npgL[0][kk, jj, ii]
                    if (npgC[kk, jj, ii  ]>-1): colG[npgL[0][kk, jj, ii]*2+0] = npgC[kk, jj, ii  ]; dataG[npgL[0][kk, jj, ii]*2+0] = 1.0
                    if (npgC[kk, jj, ii-1]>-1): colG[npgL[0][kk, jj, ii]*2+1] = npgC[kk, jj, ii-1]; dataG[npgL[0][kk, jj, ii]*2+1] = -1.0

                if (npgL[1][kk, jj, ii]>-1):
                    mat2Vl[npgL[1][kk, jj, ii]+cntL[0]] = vl_prev[kk, jj, ii, 1]*2
                    rowG[(npgL[1][kk, jj, ii]+cntL[0])*2+0] = npgL[1][kk, jj, ii]+cntL[0]
                    rowG[(npgL[1][kk, jj, ii]+cntL[0])*2+1] = npgL[1][kk, jj, ii]+cntL[0]
                    if (npgC[kk, jj,   ii]>-1): colG[(npgL[1][kk, jj, ii]+cntL[0])*2+0] = npgC[kk, jj,   ii]; dataG[(npgL[1][kk, jj, ii]+cntL[0])*2+0] = 1.0
                    if (npgC[kk, jj-1, ii]>-1): colG[(npgL[1][kk, jj, ii]+cntL[0])*2+1] = npgC[kk, jj-1, ii]; dataG[(npgL[1][kk, jj, ii]+cntL[0])*2+1] = -1.0

                if dim<3: continue
                if (npgL[2][kk, jj, ii]>-1):
                    mat2Vl[npgL[2][kk, jj, ii]+cntL[0]+cntL[1]] = vl_prev[kk, jj, ii, 2]*2
                    rowG[(npgL[1][kk, jj, ii]+cntL[0]+cntL[1])*2+0] = npgL[2][kk, jj, ii]+cntL[0]+cntL[1]
                    rowG[(npgL[1][kk, jj, ii]+cntL[0]+cntL[1])*2+1] = npgL[2][kk, jj, ii]+cntL[0]+cntL[1]
                    if (npgC[kk,   jj, ii]>-1): colG[(npgL[2][kk, jj, ii]+cntL[0]+cntL[1])*2+0] = npgC[kk,   jj, ii]; dataG[(npgL[2][kk, jj, ii]+cntL[0]+cntL[1])*2+0] = 1.0
                    if (npgC[kk-1, jj, ii]>-1): colG[(npgL[2][kk, jj, ii]+cntL[0]+cntL[1])*2+1] = npgC[kk-1, jj, ii]; dataG[(npgL[2][kk, jj, ii]+cntL[0]+cntL[1])*2+1] = -1.0

    matGrad = scipy.sparse.csr_matrix((dataG, (rowG, colG)), shape=(Ncol, Ng), dtype=np.float32)
    print('Done', flush=True)

    print('Scipy conjugate gradient solve: ', end='', flush=True)
    mat2I = scipy.sparse.identity(Ncol, dtype=np.float32)*2
    matWtWinv = scipy.sparse.linalg.inv((matW.transpose()).dot(matW) + (mat2I*beta if beta>0 else 0))
    A = matGrad.transpose().dot(matWtWinv).dot(matGrad)
    B = matGrad.transpose().dot(matWtWinv).dot(matW.transpose().dot(matVh) + (mat2Vl*beta if beta>0 else 0))

    # stats['condNo'] = scipy.sparse.linalg.norm(A)*scipy.sparse.linalg.norm(scipy.sparse.linalg.inv(A))
    X, cginfo = scipy.sparse.linalg.cg(A, B)  # X = np.linalg.solve(A, B)
    X = X.reshape((Ng, 1))
    matVl = matWtWinv.dot((matW.transpose()).dot(matVh) - matGrad.dot(X))
    print('{}'.format(cginfo), flush=True)

    corrV = np.zeros(shape=(gs[2], gs[1]+1, gs[0]+1, 3), dtype=np.float32)
    for kk in range(gs[2]):
        for jj in range(gs[1]):
            for ii in range(gs[0]):
                if npgL[0][kk, jj, ii]>-1: corrV[kk, jj, ii, 0] = matVl[npgL[0][kk, jj, ii]]
                if npgL[1][kk, jj, ii]>-1: corrV[kk, jj, ii, 1] = matVl[npgL[1][kk, jj, ii]+cntL[0]]
                if dim<3: continue
                if npgL[2][kk, jj, ii]>-1: corrV[kk, jj, ii, 2] = matVl[npgL[2][kk, jj, ii]+cntL[0]+cntL[1]]

    corr_vel = np.concatenate((corrV[..., 1:2], corrV[..., 0:1]), axis=-1)  # stack v first and u second

    return corr_vel, cginfo  # 0: successful exit, >0: convergence to tolerance not achieved, number of iterations, <0: illegal input or breakdown


class KarmanFlow(IncompressibleFlow):  # should be used for high-res
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

class KarmanFlowWithCorr(KarmanFlow):  # should be used for corrected low-res
    def __init__(self, pressure_solver=None, make_input_divfree=False, make_output_divfree=True):
        KarmanFlow.__init__(self, pressure_solver, make_input_divfree, make_output_divfree)

        self.vcorr = None

    def step(self, smoke, smokeH, solverH, re, res, velBCy, velBCyMask, dt=1.0, gravity=Gravity()):
        smoke = super().step(smoke=smoke, re=re, res=res, velBCy=velBCy, velBCyMask=velBCyMask, dt=dt, gravity=gravity)

        velocity = smoke.velocity
        density = smoke.density

        # these will be used as an input to the model
        self.den_in = CenteredGrid(density.data, smoke.density.box)
        self.vel_in = StaggeredGrid(velocity.staggered_tensor(), smoke.velocity.box)

        # correction
        self.vdiff_hi = StaggeredGrid(smokeH.velocity.staggered_tensor() - upVel(velocity.staggered_tensor()), smokeH.velocity.box)
        self.vdiff_hi = divergence_free(velocity=self.vdiff_hi, domain=smokeH.domain, obstacles=[solverH.obst], pressure_solver=solverH.pressure_solver)

        self.vcorr_prev = self.vcorr if self.vcorr is not None else StaggeredGrid(np.zeros(smoke.velocity.staggered_tensor().shape), smoke.velocity.box)
        self.vcorr, cginfo = solveVCorrLMopt(corr_prev=self.vcorr_prev.staggered_tensor(), dens_lo=smoke.density.data, dens_hi=smokeH.density.data, velo_hi=self.vdiff_hi.staggered_tensor(), beta=(params['beta']/dt))
        self.vcorr = StaggeredGrid(self.vcorr, smoke.velocity.box)

        return smoke.copied_with(density=density, velocity=velocity+self.vcorr)

st_hi = Fluid(Domain(resolution=[params['scale']*params['res']*2, params['scale']*params['res']], box=box[0:params['len']*2, 0:params['len']], boundaries=OPEN), buoyancy_factor=0)
st_co = Fluid(Domain(resolution=[params['res']*2, params['res']], box=box[0:params['len']*2, 0:params['len']], boundaries=OPEN), buoyancy_factor=0)

# init density & velocity

dens_cen = CenteredGrid(st_hi.density.data, st_hi.density.box)

vn_hi = st_hi.velocity.staggered_tensor()
vn_hi[..., 0] = 1.0       # warm start - initialize flow to 1 along y everywhere
vn_hi[..., vn_hi.shape[1]//2+10:vn_hi.shape[1]//2+20, vn_hi.shape[2]//2-2:vn_hi.shape[2]//2+2, 1] = 1.0  # modify x, poke sideways to trigger instability
vel_smac = StaggeredGrid(unstack_staggered_tensor(vn_hi), st_hi.velocity.box)

st_hi = st_hi.copied_with(density=dens_cen, velocity=vel_smac)
st_co = st_co.copied_with(density=downsample4x(dens_cen), velocity=downsample4xSMAC(vel_smac.staggered_tensor()))

# velocity BC
vn_hi = np.zeros(st_hi.velocity.data[0].data.shape)  # st_hi.velocity.data[0] is considered as the velocity field in y axis!
vn_hi[..., 0:2, 0:vn_hi.shape[2]-1, 0] = 1.0
vn_hi[..., 0:vn_hi.shape[1], 0:1,   0] = 1.0
vn_hi[..., 0:vn_hi.shape[1], -1:,   0] = 1.0
velBCy_hi = vn_hi
velBCyMask_hi = np.copy(vn_hi)  # warning, only works for 1s, otherwise setup/scale

vn_co = np.zeros(st_co.velocity.data[0].data.shape)  # st_co.velocity.data[0] is considered as the velocity field in y axis!
vn_co[..., 0:2, 0:vn_co.shape[2]-1, 0] = 1.0
vn_co[..., 0:vn_co.shape[1], 0:1,   0] = 1.0
vn_co[..., 0:vn_co.shape[1], -1:,   0] = 1.0
velBCy_co = vn_co
velBCyMask_co = np.copy(vn_co)  # warning, only works for 1s, otherwise setup/scale

# phiflow scene

scene = Scene.create(directory=params['output'])

log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))
log.info(params)

if params['output']:
    with open(os.path.normpath(scene.path)+'/params.pickle', 'wb') as f: pickle.dump(params, f)

simulator_hi = KarmanFlow()
simulator_co = KarmanFlowWithCorr()

for i in range(1, params['simsteps']):
    st_hi = simulator_hi.step(st_hi, re=params['re'], res=params['scale']*params['res'], velBCy=velBCy_hi, velBCyMask=velBCyMask_hi)
    st_co = simulator_co.step(st_co, st_hi, simulator_hi, re=params['re'], res=params['res'], velBCy=velBCy_co, velBCyMask=velBCyMask_co)

    log.info('Step {:06d}'.format(i))
    if params['skipsteps']<i and params['output'] is not None:
        scene.write(
            [
                st_hi.density, st_hi.velocity, st_co.density, st_co.velocity,
                simulator_co.den_in, simulator_co.vel_in, simulator_co.vcorr
            ],
            [
                'densH', 'veloH', 'densC', 'veloC',
                'dens', 'velo', 'corr'
            ],
            i
        )
        if params['thumb'] and params['output'] is not None:
            thumb_path = os.path.normpath(scene.path).replace(os.path.basename(scene.path), "thumb/{}".format(os.path.basename(scene.path)))
            distutils.dir_util.mkpath(thumb_path)
            save_img(st_hi.density.data,              10000., thumb_path + "/densH_{:06d}.png".format(i))
            save_img(st_hi.velocity.data[1].data,     10000., thumb_path + "/velUH_{:06d}.png".format(i))
            save_img(st_hi.velocity.data[0].data,     10000., thumb_path + "/velVH_{:06d}.png".format(i))
            save_img(st_co.density.data,              10000., thumb_path + "/densC_{:06d}.png".format(i))
            save_img(st_co.velocity.data[1].data,     10000., thumb_path + "/velUC_{:06d}.png".format(i))
            save_img(st_co.velocity.data[0].data,     10000., thumb_path + "/velVC_{:06d}.png".format(i))
            save_img(simulator_co.vcorr.data[1].data, 10000., thumb_path + "/corUC_{:06d}.png".format(i))
            save_img(simulator_co.vcorr.data[0].data, 10000., thumb_path + "/corVV_{:06d}.png".format(i))
