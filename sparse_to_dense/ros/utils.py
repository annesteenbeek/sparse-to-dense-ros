import numpy as np
import dataloaders.transforms as transforms
from scipy.spatial import ConvexHull, cKDTree
from sparse_to_dense.msg import Result as ResultMsg
from sensor_msgs.msg import Image, CameraInfo
from PIL import Image as PILImage
from PIL import ImageDraw
from cv_bridge import CvBridge, CvBridgeError

def get_result_msg(result, count=1):
    msg = ResultMsg()

    msg.irmse = result.irmse
    msg.imae = result.imae
    msg.mse = result.mse
    msg.rmse = result.rmse
    msg.mae  = result.mae 
    msg.absrel = result.absrel
    msg.lg10 = result.lg10
    msg.delta1 = result.delta1
    msg.delta2 = result.delta2
    msg.delta3 = result.delta3
    msg.data_time = result.data_time
    msg.margin10 = result.margin10
    msg.filtered = result.filtered
    msg.gpu_time = result.gpu_time
    msg.count = count

    return msg

def get_camera_info_msg(iheight, iwidth, oheight, owidth, use_tello):
    """ Generates a camera info message, and calculates the new camera
    parameters for the new info message based on resolution change.
    
    """
    # iwidth = 640
    # iheight = 480
    # iwidth = 960 # tello
    # iheight = 720
    # owidth = 304
    # oheight = 228

    camera_info_msg = CameraInfo()
    camera_info_msg.height = oheight
    camera_info_msg.width = owidth

    if not use_tello:
        # kinect params
        fx, fy = 525, 525
        cx, cy = 319.5, 239.5
    else:
        # tello params
        fx, fy = 922.93, 926.02
        cx, cy = 472.10, 384.04

    ratiox = owidth / float(iwidth)
    ratioy = oheight / float(iheight)
    fx *= ratiox
    fy *= ratioy
    cx *= ratiox
    cy *= ratioy

    camera_info_msg.K = [fx, 0, cx,
                    0, fy, cy,
                    0, 0, 1]
                        
    camera_info_msg.D = [0, 0, 0, 0]

    camera_info_msg.P = [fx, 0, cx, 0,
                        0, fy, cy, 0,
                        0, 0, 1, 0]

    return camera_info_msg

cvBridge = CvBridge()
def val_transform(img_msg, oheight, owidth):
    img_cv = cvBridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

    resize = 240.0/480
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop((oheight, owidth)),
    ])

    if img_cv.ndim == 2: # depth image
        n_depth = np.count_nonzero(img_cv)
        if n_depth < 1000: # if less then n points, must be sparse
            # to prevent sparse point loss, dont use normal resize

            def sparse_resize(img):
                rb, cb = img_cv.shape # big
                rs, cs = int(rb*resize), int(cb*resize) # small
                sh = (rs, int(rb/rs), cs, int(cb/cs)) # new shape 
                return img_cv.reshape(sh).max(-1).max(1)

            transform = transforms.Compose([
                sparse_resize,
                transforms.CenterCrop((oheight, owidth)),
            ])

    img_np = transform(img_cv)
    if img_cv.ndim == 3: # rgb images to floats
        img_np = np.asfarray(img_np, dtype='float') / 255
    # else:
    #     if n_depth < 1000:
    #         n_new = np.count_nonzero(img_np)
    #         print("Sparse point loss, before: {}, after: {}, loss: {} |".format(
    #             n_depth, n_new, n_depth-n_new
    #         ))

    return img_np

def convex_mask(sparse_depth):
    r, c= sparse_depth.shape

    zr, zc = np.nonzero(sparse_depth)

    if len(zr) < 2:
        return sparse_depth

    points = np.stack((zc,zr), axis=-1)
    hull = ConvexHull(points)
    hull_vertice_points = hull.points[hull.vertices].flatten().tolist()

    img = PILImage.new('L', (c,r), 0)
    ImageDraw.Draw(img).polygon(hull_vertice_points, outline=1, fill=1)
    depth_mask = np.array(img, dtype=bool)

    return depth_mask

def region_mask(sparse_depth):
        r = 10
        out_shp = sparse_depth.shape
        X,Y = [np.arange(-r,r+1)]*2
        disk_mask = X[:,None]**2 + Y**2 <= r*r
        Ridx,Cidx = np.where(disk_mask)

        mask = np.zeros(out_shp,dtype=bool)

        maskcenters = np.stack(np.nonzero(sparse_depth), axis=-1)
        absidxR = maskcenters[:,None,0] + Ridx-r
        absidxC = maskcenters[:,None,1] + Cidx-r

        valid_mask = (absidxR >=0) & (absidxR <out_shp[0]) & \
                    (absidxC >=0) & (absidxC <out_shp[1])

        mask[absidxR[valid_mask],absidxC[valid_mask]] = 1

        return mask 

def statistical_outlier_removal(depth_np, k=40, std_mul=0.2):
    # kinect params
    ratio = 0.475
    fx, fy = 525., 525.
    cx, cy = 319.5, 239.5
    
    fx *= ratio
    fy *= ratio
    cx *= ratio
    cy *= ratio

    # convert pixels to world space
    v, u = np.nonzero(depth_np)

    z = depth_np[v,u]
    x = (u - cx) * z / fx # x is columns
    y = (v - cy) * z / fy # y is rows

    # create KDTree
    data = zip(x.ravel(), y.ravel(), z.ravel())
    tree = cKDTree(data)
    distances, _ = tree.query(data, k)
    # remove self from matches
    distances = distances[:,1::]

    _mean = np.mean(distances)
    _std = np.std(distances)

    mean_pp = np.mean(distances, axis=1)
    threshold = _mean + std_mul * _std
    vmask = mean_pp>threshold

    rem_mask = np.zeros(depth_np.shape, dtype=bool)

    rem_mask[v[vmask], u[vmask]] = True
    return rem_mask

def median_filter(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    #     return data[s<m]
    return s>m   
