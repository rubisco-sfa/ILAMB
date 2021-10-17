from re import L
import numpy as np
from scipy.stats import linregress
import pandas as pd
from copy import deepcopy
from time import time


def gen_test(*args):
    """ Generate data with a linear trend

    Parameters
    ----------
    *args : tuple of integers
        The size of the generated array along each dimension.
    """
    np.random.seed(500)
    if len(args) == 1:
        data = 3 + 0.005 * np.arange(args[0]) + np.random.randn(args[0])
    else:
        data = 3 + 0.005 * np.broadcast_to(np.arange(args[0]).reshape(-1, *((1,)*(len(args)-1))), args) + \
            np.random.randn(*args)
    return data


def _ols_tensor(Y, x):
    """ Repeated calculation of linear regression in the spatial dimensions.

    Parameters
    ----------
    Y : np.ma.array
        The variable of interest. The first dimension will be assumed to be
        time (replicate observations).
    x : np.array or np.ma.array
        The time variable of interest. If one-dimensional, will be propagated
        to the dimensionality of Y. If having the same dimensionality as Y,
        must be a masked array.

    Returns
    -------
    r : np.ma.array
        The trend. If x only has a time dimension, `r` is a scalar.
        Otherwise, `r` has the same dimensionality as x[1:].
    p : np.ma.array
        The two-sided p-values of the trend. If x only has a time 
        dimension, `p` is a scalar. Otherwise, `p` has the same 
        dimensionality as x[1:].
    """
    if type(Y) != np.ma.core.MaskedArray:
        raise TypeError('Y must be a masked array')
    if Y.shape[0] < 3:
        raise ValueError('At least three observations are needed')

    if (type(x) != np.ma.core.MaskedArray) and (type(x) != np.ndarray):
        raise TypeError('x must be either masked or ordinary numpy array')
    if (not np.allclose(x.shape, Y.shape)) and (len(x.shape) != 1):
        raise ValueError('x must be either 1-dimensional or has the same shape as Y')

    # homogenize the shape and mask of x and Y
    if type(Y.mask) == bool:
        Y.mask = np.full(Y.shape, Y.mask)
    if type(x) == np.ma.core.MaskedArray:
        if type(x.mask) == bool:
            x.mask = np.full(x.shape, x.mask)
    else:
        x = np.ma.array(x, mask = np.full(x.shape, False))

    orig_shape = Y.shape
    Y = Y.reshape(Y.shape[0], 1, int(np.prod(Y.shape[1:])))
    if len(x.shape) != 1:
        x = x.reshape(Y.shape)
    else:
        x = np.ma.array(np.broadcast_to(x.data.reshape(-1,1,1), Y.shape),
                        mask = np.broadcast_to(x.mask.reshape(-1,1,1), Y.shape))
    x = np.ma.array(x.data, mask = x.mask | Y.mask)
    Y = np.ma.array(Y, mask = x.mask)

    # add constant term
    x = np.ma.concatenate([np.ma.array(np.ones(Y.shape), mask = Y.mask), x], axis = 1)

    # calculate the regression coefficients; treating the masked points as if zero.
    xx = np.where(x.mask == False, x.data, 0.)
    yy = np.where(Y.mask == False, Y.data, 0.)
    beta = np.einsum('ijk,jlk->ilk',
                     np.einsum('ijk,ljk->ilk',
                               np.linalg.pinv(np.einsum('ijk,ilk->jlk',xx,xx).transpose(2,0,1) \
                                             ).transpose(1,2,0),
                               xx), yy)

    # calculate the p-value
    from scipy.stats import t
    dof = np.sum(Y.mask == False, axis = 0) - 2
    resid = yy - np.einsum('ijk,jlk->ilk', xx, beta)
    mse = np.sum(np.power(resid,2), axis=0) / dof
    std = np.ma.sum(np.ma.power(x[:,[1],:] - np.ma.mean(x[:,[1],:],axis=0,keepdims=True), 2), axis = 0)
    tval = beta / np.sqrt(mse/std)
    pval = 2 * t.sf(np.abs(tval), dof)

    # discard intercept & restore shape
    beta = np.ma.array(beta[1,:], mask = np.sum(Y.mask==False, axis = 0)<3)
    pval = np.ma.array(pval[1,:], mask = np.sum(Y.mask==False, axis = 0)<3)
    if len(orig_shape) > 1:
        beta = beta.reshape(orig_shape[1:])
        pval = pval.reshape(orig_shape[1:])
    else:
        beta = float(beta.data)
        pval = float(pval.data)
    return beta, pval


if __name__ == '__main__':
    # DEBUG trend calculation
    # (1) 1-D case
    data = gen_test(100)
    data = np.ma.array(data, mask = False)

    beta, pval = _ols_tensor(data, np.arange(100))
    res = linregress(np.arange(100), data)
    print('Test 1')
    print(beta, res.slope)
    print(pval, res.pvalue)

    # (2) 2-D case
    # (2.1) Without missing values
    data = gen_test(50, 3)
    data = np.ma.array(data, mask = False)

    beta, pval = _ols_tensor(data, np.arange(50))
    which = 2
    res = linregress(np.arange(50), data[:, which])
    print('Test 2.1')
    print(beta[which], res.slope)
    print(pval[which], res.pvalue)

    # (2.2) With missing values
    data = gen_test(50, 3)
    mask = np.full(data.shape, False)
    mask[0,1:3] = True
    mask[:,2] = True
    data = np.ma.array(data, mask = mask)
    # print(data) # DEBUG
    # (2.2.1) 1-D x
    t = np.arange(50)

    beta, pval = _ols_tensor(data, np.arange(50))
    which = 1
    x = np.arange(50)
    x = x[data.mask[:,which] == False]
    res = linregress(x, data.data[data.mask[:,which] == False, which])
    print('Test 2.2.1')
    print(beta[which], res.slope)
    print(pval[which], res.pvalue)

    # (2.2.2) 2-D x
    mask2 = np.full(data.shape, False)
    mask2[0,1:3] = True
    mask2[:,2] = True
    t = np.ma.array(np.stack([np.arange(50), np.arange(50),
                              np.arange(50)], axis = 1),
                    mask = mask2)
    beta, pval = _ols_tensor(data, t)
    which = 1
    x = np.arange(50)
    x = x[data.mask[:,which] == False]
    res = linregress(x, data.data[data.mask[:,which] == False, which])
    print('Test 2.2.2')
    print(beta[which], res.slope)
    print(pval[which], res.pvalue)

    # (3) 3-D case
    data = gen_test(50, 300, 500)
    mask = np.full(data.shape, False)
    mask[0,1:3,4:6] = True
    mask[:,2,4] = True
    data = np.ma.array(data, mask = mask)
    # (3.1) 1-D x
    beta, pval = _ols_tensor(data, np.arange(50))
    which = slice(None), 2, 5
    x = np.arange(50)
    y = data.data[which]
    y[data.mask[which]] = np.nan
    res = linregress(x[~np.isnan(y)], y[~np.isnan(y)])
    print('Test 3.1')
    print(beta[which[1:]], res.slope)
    print(pval[which[1:]], res.pvalue)

    # (3.2) 3-D x
    t = np.broadcast_to(np.arange(50).reshape(50,1,1),
                        data.shape)
    beta, pval = _ols_tensor(data, t)
    which = slice(None), 2, 5
    x = np.arange(50)
    y = data.data[which]
    y[data.mask[which]] = np.nan
    res = linregress(x[~np.isnan(y)], y[~np.isnan(y)])
    print('Test 3.2')
    print(beta[which[1:]], res.slope)
    print(pval[which[1:]], res.pvalue)

    # ----- DEBUG restore shape
    #x0, x, y0, y = partial_corr_tensor(x, y, [c1, c2])
    #print(x0.data, x.data)
    #print(y0.data, y.data)
