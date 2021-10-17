import numpy as np
from numpy.core.arrayprint import array2string
import pingouin as pg
import pandas as pd
from copy import deepcopy

# Generate the randomly correlated data
def gen_test(n):
    size = 4

    def gen_cov():
        np.random.seed(800)
        cov = np.sqrt(np.random.rand(size, size)) * (2*(np.random.rand(size, size) > 0.5 - 0.5))
        cov = cov@cov.T
        return cov
    cov = gen_cov()
    # print(cov) # DEBUG

    data = np.random.multivariate_normal(np.arange(size), cov, n)

    data = pd.DataFrame(data, index = np.arange(n),
                        columns = ['x', 'y', 'c1', 'c2'])

    return data


# Test calculating the partial correlation in vectorized manner
def _covar_tensor(tensor3d):
    N = tensor3d.shape[0]
    m1 = tensor3d - tensor3d.sum(0, keepdims=True) / N
    y_out = np.einsum('ijk,ilk->jlk',m1,m1)/(N-1)
    return y_out


def partial_corr_tensor(x, y, covar_list):
    """ Repeated calculation of partial correlation in the spatial dimensions.

    Parameters
    ----------
    x : np.ma.array
        The independent variable. The first dimension will be assumed to be
        time (replicate observations). 
    y : np.ma.array
        The dependent variable. y must have the same dimensionality as x.
    covar_list : list of np.ma.array objects
        The covariate variables. Each variable must have the same
        dimensionality as x and y.

    Returns
    -------
    r : np.ma.array
        The partial correlation. If x only has a time dimension, `r` is a
        a scalar. Otherwise, `r` has the same dimensionality as x[1:].
    p : np.ma.array
        The two-sided p-values of the partial correlation. If x only has 
        a time dimension, `p` is a scalar. Otherwise, `p` has the same 
        dimensionality as x[1:].
    """
    if type(x) != np.ma.core.MaskedArray:
        raise TypeError('x must be a masked array')
    if type(y) != np.ma.core.MaskedArray:
        raise TypeError('y must be a masked array')
    for vv in covar_list:
        if type(vv) != np.ma.core.MaskedArray:
            raise ValueError('covar_list must be masked arrays')
    if not np.allclose(x.shape, y.shape):
        raise ValueError('x and y must be the same shape')
    for vv in covar_list:
        if not np.allclose(x.shape, vv.shape):
            raise ValueError('x and covar_list must be the same shape')
    if x.shape[0] < 3:
        raise ValueError('At least three observations are needed')

    x0 = x.copy()
    y0 = y.copy()

    orig_shape = x.shape
    if len(orig_shape) == 1:
        x = x.reshape(-1, 1, 1) # extra 2nd dimension for concat
        y = y.reshape(-1, 1, 1)
        covar_relist = []
        for vv in covar_list:
            covar_relist.append(vv.reshape(-1, 1, 1))
    else:
        new_shape = orig_shape[0], 1, np.prod(orig_shape[1:])
        x = x.reshape(new_shape)
        y = y.reshape(new_shape)
        covar_relist = []
        for vv in covar_list:
            covar_relist.append(vv.reshape(new_shape))
    covar_list = covar_relist
    covar_relist = []

    data = np.ma.concatenate([x,y] + covar_list, axis = 1)
    del x, y
    covar_list = []

    # remove invalid points
    retain_ind = np.any(np.all(data.mask == False, axis = 1), axis = 0)
    if sum(retain_ind) == 0:
        raise ValueError('At least one valid spatial data point is needed')
    ## print(retain_ind) # DEBUG
    data = data[:, :, retain_ind]

    # TO-DO: Need to think of a way to deal with the different number
    #        of data points in space. Right now it imposes the minimum
    #        overlapping number of valid data points.
    drop_replica = np.all(np.all(data.mask == False, axis = 2), axis = 1)
    ## print(drop_replica) # DEBUG
    if sum(drop_replica) < 3:
        raise ValueError('At least three valid observations are needed')
    data = data[drop_replica, :, :]

    # calculate the partial correlation and significance (translated from pingouin)
    V = _covar_tensor(data)
    ##print(data.shape) # DEBUG
    ##print(V.shape) # DEBUG

    Vi = np.linalg.inv(V.transpose(2,0,1)).transpose(1,2,0)
    D = np.zeros(Vi.shape)
    for ii in np.arange(Vi.shape[0]):
        D[ii,ii,:] = np.sqrt( 1 / Vi[ii,ii,:] )
    pcor = -1 * np.einsum('jik,ilk->jlk', np.einsum('jik,ilk->jlk',D,Vi), D)
    ## print(-1 * D[:,:,5] @ Vi[:,:,5] @ D[:,:,5] - pcor[:,:,5]) # check if correct
    r = pcor[0, 1, :]

    from scipy.stats import t
    n = data.shape[0]
    k = data.shape[1] - 2
    dof = n - k - 2
    tval = r * np.sqrt(dof / (1 - r**2))
    pval = 2 * t.sf(np.abs(tval), dof)

    # restore shape
    def _restore_shape(array, retain_ind, orig_shape):
        array_restore = np.ma.empty(len(retain_ind))
        array_restore.mask = retain_ind == False
        array_restore.data[retain_ind] = array
        array_restore = array_restore.reshape(orig_shape[1:])
        return array_restore

    # DEBUG restore shape
    ##return x0[drop_replica,:][0,:], _restore_shape(data[0,0,:], retain_ind, orig_shape), \
    ##    y0[drop_replica,:][0,:], _restore_shape(data[0,1,:], retain_ind, orig_shape)

    if len(orig_shape) == 1:
        return r[0], pval[0]
    else:
        r_restore = _restore_shape(r, retain_ind, orig_shape)
        p_restore = _restore_shape(pval, retain_ind, orig_shape)
        return r_restore, p_restore


if __name__ == '__main__':

    # DEBUG covariance calculation
    """
    data = gen_test(100)
    print(data) # DEBUG
    print( np.cov(data.values.T) )
    print( _covar_tensor(data.values.reshape(*data.shape, 1))[:, :, 0] )
    """

    # DEBUG partial correlation calculation
    data = gen_test(500)
    data.iloc[5:60, :] = np.nan
    x = np.ma.masked_where(np.isnan(data['x'].values),
                           data['x'].values).reshape(25, 20)
    y = np.ma.masked_where(np.isnan(data['y'].values),
                           data['y'].values).reshape(25, 20)
    c1 = np.ma.masked_where(np.isnan(data['c1'].values),
                            data['c1'].values).reshape(25, 20)
    c2 = np.ma.masked_where(np.isnan(data['c2'].values),
                            data['c2'].values).reshape(25, 20)

    # ----- DEBUG restore shape
    #x0, x, y0, y = partial_corr_tensor(x, y, [c1, c2])
    #print(x0.data, x.data)
    #print(y0.data, y.data)

    # ----- DEBUG the actual partial correlation
    which = 6
    stats = pg.partial_corr(data.iloc[which::20, :].dropna(axis = 1, how = 'all'), x = 'x', y = 'y', covar = ['c1', 'c2'])
    print(stats)

    # Tensor-calculate the partial correlation
    r, p = partial_corr_tensor(x, y, [c1, c2])
    print(r[which])
    print(p[which])
    print(r)
    print(p)
