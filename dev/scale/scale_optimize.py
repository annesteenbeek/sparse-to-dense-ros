
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


sparse_points = np.load('dev/scale/sparse_points.npy')
est_points = np.load('dev/scale/est_points.npy')
targ_points = np.load('dev/scale/targ_points.npy')
n_points = np.load('dev/scale/n_depths.npy')

n_points = n_points.astype(int)

n_total = np.sum(n_points)

error = targ_points - sparse_points
scale = np.median(targ_points/sparse_points)


def _scale_error(scale):
    error = np.abs(scale*sparse_points - est_points)**2
    return error.sum()

def _true_scale(scale):
    error = np.abs(scale*sparse_points - targ_points)**2
    return error.sum()

def _test_scale(scale):
    error = np.abs(scale*sparse_points - targ_points)
    return error.mean()

min_scale = minimize_scalar(_scale_error)
min_targ_scale = minimize_scalar(_true_scale)

print("old scale: %.3f, mean error: %.3f" % (scale, _test_scale(scale)))
print("est scale: %.3f, mean error: %.3f" % (min_scale.x, _test_scale(min_scale.x)))
print("targ scale: %.3f, mean error: %.3f" % (min_targ_scale.x, _test_scale(min_targ_scale.x)))


# on a per frame basis

def _frame_scale_error(scale, sparse, est):
    error = np.abs(scale*sparse - est)**2
    return error.sum()

n_prev = 0
frame_scales = []
for n in n_points:
    spar = sparse_points[n_prev:n]
    est = est_points[n_prev:n]

    frame_scale = minimize_scalar(_frame_scale_error, args=(spar, est))
    print(frame_scale.x)
    frame_scales.append(frame_scale.x)

    n_prev += n
print(np.median(frame_scales))
    
