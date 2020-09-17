# ----------------------------------------------------------------------------
#
# Phiflow Burgers equation solver framework
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
parser.add_argument('-t', '--simsteps',  default=200, type=int,   help='simulation steps')
parser.add_argument('-r', '--res',       default=32, type=int,    help='resolution of the reference axis')
parser.add_argument('-l', '--len',       default=96, type=int,    help='length of the reference axis')
parser.add_argument('--dt',              default=1.0, type=float, help='simulation time step size')
parser.add_argument('--initvH',          default=None,            help='load hires (will be downsampled) velocity (e.g., velo_0000.npz)')
parser.add_argument('--loadfH',          default=None,            help='load hires (will be downsampled) force files (will be passed to glob) (e.g., "sim_000000/forc_0*.npz")')
parser.add_argument('-d', '--scale',     default=4, type=int,     help='down-sampling scale of hires (only valid when initvH given)')
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

def downsample4xSMAC(tensor):
    return StaggeredGrid(tensor).downsample2x().downsample2x().staggered_tensor()

downVel = eval('downsample{}xSMAC'.format(params['scale']))
upVel = eval('upsample{}xSMAC'.format(params['scale']))

def fluidCellIndexes(tensor_cen, bnd):
    # just fill all cells except boundary (bnd) for now, may consider density values afterwards?
    cnt, npg = 0, np.ones(shape=(1, tensor_cen.shape[-3], tensor_cen.shape[-2]), dtype=np.int32)*-1

    for j in np.arange(bnd, tensor_cen.shape[-3]-bnd):
        for i in np.arange(bnd, tensor_cen.shape[-2]-bnd):
            npg[0, j, i] = cnt
            cnt += 1

    return cnt, npg

def magnifyCellIndexes(src_cen, scale):
    cnt, npg = 0, np.ones(shape=(1, src_cen.shape[1]*scale, src_cen.shape[2]*scale), dtype=np.int32)*-1

    for j in np.arange(npg.shape[1]):
        for i in np.arange(npg.shape[2]):
            if src_cen[0, j//scale, i//scale]>-1:
                npg[0, j, i] = cnt
                cnt += 1

    return cnt, npg

def fluidFaceIndexes(cen_index, dim, bnd):
    # if at least one adjacent cell has a valid index
    cnt, npg = [0]*dim, [np.ones(shape=cen_index.shape, dtype=np.int32)*-1 for _ in range(dim)]

    for k in (np.arange(bnd, cen_index.shape[-3]-bnd) if dim>2 else np.arange(1)):
        for j in np.arange(bnd, cen_index.shape[-2]-bnd):
            for i in np.arange(bnd, cen_index.shape[-1]-bnd):
                if (cen_index[k, j, i]>-1) or (i>0 and cen_index[k, j, i-1]>-1):
                    npg[0][k, j, i] = cnt[0]
                    cnt[0] += 1

                if (cen_index[k, j, i]>-1) or (j>0 and cen_index[k, j-1, i]>-1):
                    npg[1][k, j, i] = cnt[1]
                    cnt[1] += 1

                if dim<3: continue
                if (cen_index[k, j, i]>-1) or (k>0 and cen_index[k-1, j, i]>-1):
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
    NgH, npgCH = magnifyCellIndexes(src_cen=npgC, scale=params['scale'])

    cntL, npgL = fluidFaceIndexes(cen_index=npgC,  dim=dim, bnd=0)
    cntH, npgH = fluidFaceIndexes(cen_index=npgCH, dim=dim, bnd=0)
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
    A = matW.transpose().dot(matW) + (mat2I*beta if beta>0 else 0)
    B = matW.transpose().dot(matVh) + (mat2Vl*beta if beta>0 else 0)

    # stats['condNo'] = scipy.sparse.linalg.norm(A)*scipy.sparse.linalg.norm(scipy.sparse.linalg.inv(A))
    X, cginfo = scipy.sparse.linalg.cg(A, B)  # X = np.linalg.solve(A, B)
    matVl = X
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

class BurgersCorr(BurgersTest):
    def __init__(self, st, default_viscosity=0.1, viscosity=None, diffusion_substeps=1):
        BurgersTest.__init__(self, default_viscosity=default_viscosity, viscosity=viscosity, diffusion_substeps=diffusion_substeps)

        self.vcorr  = StaggeredGrid(np.zeros(st.velocity.staggered_tensor().shape), st.velocity.box)
        self.vel_in = StaggeredGrid(np.zeros(st.velocity.staggered_tensor().shape), st.velocity.box)

    def step_with_f(self, v, f, stH, dt=1.0):
        st = super().step_with_f(v=v, f=f, dt=dt)

        # dummy centered grids
        density  = st.centered_grid(name="dummy_d", value=1.0)
        densityH = stH.centered_grid(name="dummy_dH", value=1.0)

        # this will be used as an input to the model
        self.vel_in = StaggeredGrid(st.velocity.staggered_tensor(), st.velocity.box)

        # correction
        self.vdiff_hi = StaggeredGrid(stH.velocity.staggered_tensor() - upVel(st.velocity.staggered_tensor()), stH.velocity.box)

        self.vcorr_prev = self.vcorr.copied_with()
        self.vcorr, cginfo = solveVCorrLMopt(corr_prev=self.vcorr_prev.staggered_tensor(), dens_lo=density.data, dens_hi=densityH.data, velo_hi=self.vdiff_hi.staggered_tensor(), beta=(params['beta']/dt))
        self.vcorr = StaggeredGrid(self.vcorr, st.velocity.box)

        return st.copied_with(velocity=st.velocity+self.vcorr)


dm_hi = Domain(resolution=[params['scale']*params['res']]*2, box=box([params['len']]*2), boundaries=PERIODIC)
dm_co = Domain(resolution=[                params['res']]*2, box=box([params['len']]*2), boundaries=PERIODIC)

# init density & velocity

st_hi = BurgersVelocitySMAC(dm_hi, velocity=lambda s: math.randfreq(s) * 2)
st_co = BurgersVelocitySMAC(dm_co, velocity=downVel(st_hi.velocity.staggered_tensor()))

if params['initvH']:
    st_hi = st_hi.copied_with(velocity=read_zipped_array(params['initvH']))
    st_co = st_co.copied_with(velocity=downVel(st_hi.velocity.staggered_tensor()))

fc_files = sorted(glob.glob(params['loadfH'])) if params['loadfH'] else None
fc_hi = st_hi.copied_with(velocity=        read_zipped_array(fc_files[0]))
fc_co = st_co.copied_with(velocity=downVel(read_zipped_array(fc_files[0])))

# phiflow scene

scene = Scene.create(directory=params['output'])

log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))
log.info(params)

if params['output']:
    with open(os.path.normpath(scene.path)+'/params.pickle', 'wb') as f: pickle.dump(params, f)

simulator_hi = BurgersTest()
simulator_co = BurgersCorr(st_co)

if params['output'] is not None:
    scene.write(
        [
            st_hi.velocity, st_co.velocity,
            simulator_co.vel_in, simulator_co.vcorr,
            fc_hi.velocity, fc_co.velocity,
        ],
        [
            'veloH', 'veloC',
            'velo',  'corr',
            'forcH', 'forc',
        ],
        0
    )
    if params['thumb']:
        thumb_path = os.path.normpath(scene.path).replace(os.path.basename(scene.path), "thumb/{}".format(os.path.basename(scene.path)))
        distutils.dir_util.mkpath(thumb_path)
        save_img(st_hi.velocity.data[1].data,     100000., thumb_path + "/velUH_{:06d}.png".format(0))
        save_img(st_hi.velocity.data[0].data,     100000., thumb_path + "/velVH_{:06d}.png".format(0))
        save_img(st_co.velocity.data[1].data,     100000., thumb_path + "/velUC_{:06d}.png".format(0))
        save_img(st_co.velocity.data[0].data,     100000., thumb_path + "/velVC_{:06d}.png".format(0))
        save_img(fc_hi.velocity.data[1].data,     100000., thumb_path + "/forcH_{:06d}.png".format(0))
        save_img(fc_hi.velocity.data[0].data,     100000., thumb_path + "/forcH_{:06d}.png".format(0))
        save_img(fc_co.velocity.data[1].data,     100000., thumb_path + "/forcC_{:06d}.png".format(0))
        save_img(fc_co.velocity.data[0].data,     100000., thumb_path + "/forcC_{:06d}.png".format(0))
        save_img(simulator_co.vcorr.data[1].data, 100000., thumb_path + "/corUC_{:06d}.png".format(0))
        save_img(simulator_co.vcorr.data[0].data, 100000., thumb_path + "/corVV_{:06d}.png".format(0))

for i in range(1, params['simsteps']):
    st_hi = simulator_hi.step_with_f(v=st_hi, f=fc_hi, dt=params['dt'])
    st_co = simulator_co.step_with_f(v=st_co, f=fc_co, stH=st_hi, dt=params['dt'])

    fc_hi = fc_hi.copied_with(velocity=        read_zipped_array(fc_files[i]))
    fc_co = fc_co.copied_with(velocity=downVel(read_zipped_array(fc_files[i])))

    log.info('Step {:06d}'.format(i))
    if params['output'] is not None:
        scene.write(
            [
                st_hi.velocity, st_co.velocity,
                simulator_co.vel_in, simulator_co.vcorr,
                fc_hi.velocity, fc_co.velocity,
            ],
            [
                'veloH', 'veloC',
                'velo', 'corr',
                'forcH', 'forc',
            ],
            i
        )
        if params['thumb'] and params['output'] is not None:
            thumb_path = os.path.normpath(scene.path).replace(os.path.basename(scene.path), "thumb/{}".format(os.path.basename(scene.path)))
            distutils.dir_util.mkpath(thumb_path)
            save_img(st_hi.velocity.data[1].data,     100000., thumb_path + "/velUH_{:06d}.png".format(i))
            save_img(st_hi.velocity.data[0].data,     100000., thumb_path + "/velVH_{:06d}.png".format(i))
            save_img(st_co.velocity.data[1].data,     100000., thumb_path + "/velUC_{:06d}.png".format(i))
            save_img(st_co.velocity.data[0].data,     100000., thumb_path + "/velVC_{:06d}.png".format(i))
            save_img(fc_hi.velocity.data[1].data,     100000., thumb_path + "/forcH_{:06d}.png".format(i))
            save_img(fc_hi.velocity.data[0].data,     100000., thumb_path + "/forcH_{:06d}.png".format(i))
            save_img(fc_co.velocity.data[1].data,     100000., thumb_path + "/forcC_{:06d}.png".format(i))
            save_img(fc_co.velocity.data[0].data,     100000., thumb_path + "/forcC_{:06d}.png".format(i))
            save_img(simulator_co.vcorr.data[1].data, 100000., thumb_path + "/corUC_{:06d}.png".format(i))
            save_img(simulator_co.vcorr.data[0].data, 100000., thumb_path + "/corVV_{:06d}.png".format(i))
