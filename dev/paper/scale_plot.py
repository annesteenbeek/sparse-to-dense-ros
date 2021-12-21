import numpy as np
from scipy.optimize import minimize_scalar
import scipy
import matplotlib.pyplot as plt


root = "scale/"


room1 = root+"fr3_long"
room2 = root+"fr2_desk"
room3 = root+"fr3_struct_text_far"
room = room2
# kinect_error = 1./1.035


sparse_points = np.load(room+'/sparse_points.npy')
est_points = np.load(room+'/est_points.npy')
targ_points = np.load(room+'/targ_points.npy')
n_points = np.load(room+'/n_depths.npy')
indexes = np.load(room+'/indexes.npy')
indexes = indexes.reshape(len(indexes)/2,2)

if room == room1:
    ref_file = room1+"/rgbd_dataset_freiburg3_long_office_household-groundtruth.txt"
    room_scale = 2.43126284678
#     targ_points = targ_points*1.035
elif room == room2:
    ref_file = room2+"/rgbd_dataset_freiburg2_desk-groundtruth.txt"
    room_scale = 2.1582600441
    targ_points = targ_points*1.031
elif room == room3:
    ref_file = room3+"/rgbd_dataset_freiburg3_structure_texture_far-groundtruth.txt"
    room_scale = 1.89191437662
#     targ_points = targ_points*1.035
else:
    raise("Unknown room")

def filter_mask(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    #     return data[s<m]
    return s>m   

def _scale_error(scale, sparse, test):
    max_dist = 3.

    scales = test/sparse
    rem_mask = test>max_dist
    
#     se = np.abs(scale*sparse[~rem_mask] - test[~rem_mask])**2
    se = np.abs(scale*sparse - test)**2
    return se.sum()

n_prev = 0
est_scales = []
est_means = []

targ_scales = []
targ_means = []

mean_dist_spar = []
mean_dist_est = []
mean_dist_targ = []

walking_scales = []
walking_means = []
for n in n_points:
    if n == 0:
        continue
    
    i = np.arange(n_prev, n_prev+n)
    j = np.arange(n_prev+n)
    spar = sparse_points[i]
    est = est_points[i]
    targ = targ_points[i]
    
    mean_dist_spar.append(np.mean(spar))
    mean_dist_est.append(np.mean(est))
    mean_dist_targ.append(np.mean(targ))
    

#     data = cam_to_xyz(sparse_points[i], indexes[i,:])
#     mask = statistical_outlier_removal(data, k=5, std_mul=1.0)

#     est_scale = minimize_scalar(_scale_error, args=(spar[~mask], est[~mask]))
#     targ_scale = minimize_scalar(_scale_error, args=(spar[~mask], targ[~mask]))
    est_scale = minimize_scalar(_scale_error, args=(spar, est))
    targ_scale = minimize_scalar(_scale_error, args=(spar, targ))
        
    est_scales.append(est_scale.x)
    est_means.append(np.mean(est_scales))
    
    targ_scales.append(targ_scale.x)
    targ_means.append(np.mean(targ_scales))
    
    # using all points up to now
    _est = est_points[j]
    _spar = sparse_points[j]
    mask = filter_mask(_est/_spar, m=0.5)
    walking_scale = minimize_scalar(_scale_error, args=(_spar[~mask], _est[~mask]))
    walking_scales.append(walking_scale.x)
    walking_means.append(np.mean(walking_scales))

    
    n_prev += n
    
print("gt scale: %.3f" % room_scale)
print("estimate mean: %.3f" % est_means[-1])
print("kinect mean: %.3f" % targ_means[-1])
print("walking end: %.3f" % walking_scales[-1])
print("over %d frames" % len(est_means))
print("average samples %.1f" % np.mean(n_points))


fig = plt.figure()
plt.style.use(['science', 'no-latex'])
plt.plot(np.arange(len(est_means)), est_means, 'b', label="CNN")
plt.plot(np.arange(len(targ_means)), targ_means, 'r', label="Kinect")
plt.plot(np.arange(len(est_scales)), est_scales, 'b--', alpha=0.3)
plt.plot(np.arange(len(targ_scales)), targ_scales, 'r--', alpha=0.3)
plt.plot(np.arange(len(walking_scales)), walking_scales, 'g', label="Median filter")

plt.title("Scale estimate")
plt.xlabel("Frame index")
plt.ylabel("Scale")
plt.ylim(0,5)
plt.axhline(linewidth=1, y=room_scale, color='k')
plt.legend()
# plt.show()

# Filter scales
est_scales = est_points/sparse_points

targ_scales = targ_points/sparse_points

# filter the scales
scale_mask = filter_mask(est_scales, m=0.5)
f_est_scales = np.copy(est_scales)
f_est_scales[scale_mask] = np.nan

# plt.figure()
# plt.title("scale estimates")
# plt.xlabel("samples")
# plt.ylabel("scale")
# plt.scatter(np.arange(len(est_scales)), est_scales, s=1, label="scales")
# plt.scatter(np.arange(len(f_est_scales)), f_est_scales, s=1, label="filtered scales")
# plt.legend()
# plt.show()

mask = filter_mask(est_points/sparse_points, m=0.5)

_est_points = np.copy(est_points)
_sparse_points = np.copy(sparse_points)
_est_points[mask] = np.nan
_sparse_points[mask] = np.nan

plt.figure()
plt.style.use(['science', 'no-latex'])
plt.title("Depth samples")
plt.scatter(np.arange(len(est_points)), sparse_points*room_scale, c='r', s=1, label="Map point")
plt.scatter(np.arange(len(est_points)), est_points, c='b', s=1, label="CNN")
plt.scatter(np.arange(len(est_points)), _sparse_points*room_scale, c='g', s=1, label="Median filter")
plt.xlabel("Map point index")
plt.ylabel("Distance [m]")
# plt.legend(frameon=True, facecolor="grey")
plt.legend(markerscale=2)
# plt.show()


est_errors = est_points - sparse_points
est_scales = est_points/sparse_points

# filter the scales
scale_mask = filter_mask(est_scales, m=0.5)
f_est_scales = np.copy(est_scales)
f_est_scales[scale_mask] = np.nan

scale_mask = filter_mask(est_scales, m=1.)
f2_est_scales = np.copy(est_scales)
f2_est_scales[scale_mask] = np.nan

scale_mask = filter_mask(est_scales, m=2.)
f3_est_scales = np.copy(est_scales)
f3_est_scales[scale_mask] = np.nan

plt.figure()
plt.title("Median filter")
plt.xlabel("scale index")
plt.ylabel("scale")
plt.scatter(np.arange(len(est_scales)), est_scales, s=1, label="Unfiltered")
plt.scatter(np.arange(len(f_est_scales)), f3_est_scales, s=1, label="f=2.0")
plt.scatter(np.arange(len(f_est_scales)), f2_est_scales, s=1, label="f=1.0")
plt.scatter(np.arange(len(f_est_scales)), f_est_scales, s=1, label="f=0.5")
plt.legend(markerscale=3)
plt.axhline(linewidth=1, y=np.mean(est_scales), color='k')
plt.show()
