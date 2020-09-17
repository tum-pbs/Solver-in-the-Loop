# ----------------------------------------------------------------------------
#
# Data manipulation
# Copyright 2019 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Data normalization
#
# ----------------------------------------------------------------------------

import numpy as np

def dataStats(idata, odata):
    return {
        'in.mean'  : [ np.mean(idata[...,i][np.nonzero(idata[...,i])]) for i in range(idata.shape[-1]) ],
        'in.std'   : [ np.std( idata[...,i][np.nonzero(idata[...,i])]) for i in range(idata.shape[-1]) ],
        'in.min'   : [ np.amin(idata[...,i]                          ) for i in range(idata.shape[-1]) ],
        'in.max'   : [ np.amax(idata[...,i]                          ) for i in range(idata.shape[-1]) ],
        'out.mean' : [ np.mean(odata[...,i][np.nonzero(odata[...,i])]) for i in range(odata.shape[-1]) ],
        'out.std'  : [ np.std( odata[...,i][np.nonzero(odata[...,i])]) for i in range(odata.shape[-1]) ],
        'out.min'  : [ np.amin(odata[...,i]                          ) for i in range(odata.shape[-1]) ],
        'out.max'  : [ np.amax(odata[...,i]                          ) for i in range(odata.shape[-1]) ],
    }

def standardize(idata, odata, dstats, sigma_range=1, zero_centered=False):
    if idata is not None:
        for i in range(idata.shape[-1]): idata[..., i] = (idata[..., i] - (0 if zero_centered else dstats['in.mean'][i]))/(sigma_range*dstats['in.std'][i])
    if odata is not None:
        for i in range(odata.shape[-1]): odata[..., i] = (odata[..., i] - (0 if zero_centered else dstats['out.mean'][i]))/(sigma_range*dstats['out.std'][i])

def deStandardize(idata, odata, dstats, sigma_range=1, zero_centered=False):
    if idata is not None:
        for i in range(idata.shape[-1]): idata[..., i] = idata[..., i]*sigma_range*dstats['in.std'][i] + (0 if zero_centered else dstats['in.mean'][i])
    if odata is not None:
        for i in range(odata.shape[-1]): odata[..., i] = odata[..., i]*sigma_range*dstats['out.std'][i] + (0 if zero_centered else dstats['out.mean'][i])

def normalize(idata, odata, dstats):
    if idata is not None:
        for i in range(idata.shape[-1]): idata[..., i] = (idata[..., i] - dstats['in.min'][i])/(dstats['in.max'][i]-dstats['in.min'][i])
    if odata is not None:
        for i in range(odata.shape[-1]): odata[..., i] = (odata[..., i] - dstats['out.min'][i])/(dstats['out.max'][i]-dstats['out.min'][i])

def deNormalize(idata, odata, dstats):
    if idata is not None:
        for i in range(idata.shape[-1]): idata[..., i] = idata[..., i]*(dstats['in.max'][i]-dstats['in.min'][i]) + dstats['in.min'][i]
    if odata is not None:
        for i in range(odata.shape[-1]): odata[..., i] = odata[..., i]*(dstats['out.max'][i]-dstats['out.min'][i]) + dstats['out.min'][i]
