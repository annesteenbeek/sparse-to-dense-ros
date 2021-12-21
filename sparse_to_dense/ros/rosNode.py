import os
import threading
import torch
import numpy as np
import skimage.transform as transform
import dataloaders.transforms as transforms
from dataloaders.dense_to_sparse import UniformSampling, ORBSampling
from metrics import AverageMeter, Result
from scipy import ndimage
from PIL import Image as PILImage
from scipy.optimize import minimize_scalar

# ROS imports
import rospy
import tf2_ros
import tf
import message_filters
import ros_numpy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from sparse_to_dense.msg import Result as ResultMsg
from sensor_msgs.msg import Image, CameraInfo
from dynamic_reconfigure.server import Server
from sparse_to_dense.cfg import SparseToDenseConfig 
from ros.sampleMetricsTracker import SampleMetricsTracker
from ros.frameSaver import FrameSaver
from ros.utils import get_result_msg, get_camera_info_msg, val_transform, convex_mask, region_mask, statistical_outlier_removal, median_filter



to_tensor = transforms.ToTensor()

class ROSNode(object):

    def __init__(self, model, oheight=228, owidth=304):
        self.model = model


        self.oheight = oheight
        self.owidth = owidth

        self.sparsifier = ORBSampling(num_samples=100, max_depth=5.0)

        self.model.eval()
        self.img_lock = threading.Lock()

        self.est_frame_saver = FrameSaver(prefix="est", enabled=True)
        self.rgb_frame_saver = FrameSaver(prefix="rgb_est", enabled=True)

        target_topic = rospy.get_param("~target_topic", "")
        self.emulate_sparse_depth = rospy.get_param("~emulate_sparse_depth", False)
        self.scale_samples = rospy.get_param("~scale_samples", 20)
        self.frame = rospy.get_param("~frame", "openni_link")
        self.max_depth = rospy.get_param("~max_depth", 100)
        self.filter_low_texture = rospy.get_param("~filter_low_texture", False)
        self.use_tello = rospy.get_param("~use_tello", False)

        self.depth_est_pub = rospy.Publisher('depth_est', Image, queue_size=5)
        self.sparse_debug_pub = rospy.Publisher('debug/sparse_depth', Image, queue_size=5)
        self.rbg_debug_pub = rospy.Publisher('debug/rgb', Image, queue_size=5)
        self.debug_cam_info_pub = rospy.Publisher('debug/camera_info', CameraInfo, queue_size=5)
        self.cam_info_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=5)
        self.avg_res_pub = rospy.Publisher('average_results', ResultMsg, queue_size=5)
        self.sample_metrics_tracker = SampleMetricsTracker(self.max_depth)

        self.rgb_sub = message_filters.Subscriber('rgb_in', Image)
        self.sparse_sub = message_filters.Subscriber('depth_sparse', Image)
        if target_topic == "":
            self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.sparse_sub], queue_size=20, slop=0.02)
        else:
            self.target_debug_pub = rospy.Publisher('debug/target_depth', Image, queue_size=5)

            self.target_sub = message_filters.Subscriber(target_topic, Image)
            self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.sparse_sub, self.target_sub], queue_size=20, slop=0.02)
        self.ts.registerCallback(self.sync_img_callback)
        
        self.average_meter = AverageMeter()

        self.tf_pub = tf.TransformBroadcaster()

        self.gradient_cutoff = 0.05
        self.config_srv = Server(SparseToDenseConfig, self.config_callback)

        self.frame_nr = 0
        self.scale_est = None

        # TMP to store samples
        self.sparse_points = np.array([])
        self.est_points = np.array([])
        self.targ_points = np.array([])
        self.n_points = np.array([], dtype=int)
        self.indexes = np.array([])

        self.tfBuffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.tfBuffer)

    def config_callback(self, config, level):
        rospy.loginfo("Gradient cutoff: {gradient_cutoff}".format(**config))
        self.gradient_cutoff = config['gradient_cutoff']
        return config

    def publish_optical_transform(self, time, optical_frame):
        self.tf_pub.sendTransform((0,0,0), 
            tf.transformations.quaternion_from_euler(-np.pi/2, 0, -np.pi/2),
            time,
            optical_frame,
            self.frame)

    def publish_scaled_transform(self, time):
        if self.scale_est == None:
            rospy.logwarn_once("Tried to publish scaled transform without known scale")
            return

        try:
            orb_trans = self.tfBuffer.lookup_transform('world', 'orb_frame', time)
            scaled_translation = (orb_trans.transform.translation.x*self.scale_est,
                                    orb_trans.transform.translation.y*self.scale_est,
                                    orb_trans.transform.translation.z*self.scale_est)
            quaternion = (orb_trans.transform.rotation.x,
                            orb_trans.transform.rotation.y,
                            orb_trans.transform.rotation.z,
                            orb_trans.transform.rotation.w)

            self.tf_pub.sendTransform(scaled_translation, 
                                        quaternion,
                                        time,
                                        self.frame,
                                        'world')

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logerr("TF error: {0}".format(ex))


    def sync_img_callback(self, rgb_msg, depth_msg, target_msg=None):

        if self.img_lock.acquire(False):
            self.frame_nr+= 1
            header = rgb_msg.header 
            if self.frame_nr < self.scale_samples:
                    print("frame# %d " % self.frame_nr)
                    self.predict_scale(rgb_msg, depth_msg, target_msg)
                    self.img_lock.release()
                    return
            else:
                if self.emulate_sparse_depth:
                    # use rgbd for sparse pointcloud
                    header = target_msg.header
                    depth_pred, sparse_debug = self.emulate_predict_depth(rgb_msg, target_msg)
                else:
                    depth_pred, sparse_debug = self.predict_depth(rgb_msg, depth_msg, target_msg)

            rgb_np = val_transform(rgb_msg, self.oheight, self.owidth)
            rgb_debug_msg = ros_numpy.msgify(Image, (rgb_np*255).astype(np.uint8), encoding=rgb_msg.encoding)
            pred_msg = ros_numpy.msgify(Image, depth_pred, encoding=depth_msg.encoding)
            sparse_debug_msg = ros_numpy.msgify(Image, sparse_debug, encoding=depth_msg.encoding)


            image_frame = header.frame_id
            # image_frame = "openni_rgb_optical_frame"
            pred_msg.header.stamp = header.stamp
            pred_msg.header.frame_id = image_frame

            sparse_debug_msg.header.stamp = header.stamp
            sparse_debug_msg.header.frame_id = header.frame_id

            rgb_debug_msg.header.stamp = header.stamp
            rgb_debug_msg.header.frame_id = header.frame_id

            camera_info_msg = get_camera_info_msg(rgb_msg.height, rgb_msg.width, self.oheight, self.owidth, self.use_tello)
            camera_info_msg.header.stamp = header.stamp

            # debug_camera_info_msg = get_camera_info_msg(rgb_msg.height, rgb_msg.width, self.oheight, self.owidth, self.use_tello)
            # debug_camera_info_msg.header.stamp = header.stamp

            self.depth_est_pub.publish(pred_msg)
            self.cam_info_pub.publish(camera_info_msg)
            self.sparse_debug_pub.publish(sparse_debug_msg)
            self.rbg_debug_pub.publish(rgb_debug_msg)
            self.debug_cam_info_pub.publish(camera_info_msg)

            # for depth target pointcloud
            if target_msg is not None:
                target_np = val_transform(target_msg, self.oheight, self.owidth)
                debug_target_msg = ros_numpy.msgify(Image, target_np, encoding=depth_msg.encoding)
                # debug_target_msg.header.stamp = target_msg.header.stamp
                debug_target_msg.header.stamp = header.stamp
                debug_target_msg.header.frame_id = image_frame

                # debug_camera_info_msg = get_camera_info_msg(target_msg.height, target_msg.width, debug_target_msg.height, debug_target_msg.width)
                # debug_camera_info_msg.header.stamp = target_msg.header.stamp

                self.target_debug_pub.publish(debug_target_msg)
                self.debug_cam_info_pub.publish(camera_info_msg)

            if not self.frame_nr < self.scale_samples:
                self.publish_scaled_transform(header.stamp)

            self.publish_optical_transform(header.stamp, image_frame)

            self.img_lock.release()

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.sparsifier.dense_to_sparse(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd


    def _scale_error(self, scale, sparse, test):
        error = (scale*sparse - test)**2
        return error.sum()

    def predict_scale(self, rgb_msg, sparse_msg, target_msg=None):
        rgb_np = val_transform(rgb_msg, self.oheight, self.owidth)
        sparse_np = val_transform(sparse_msg, self.oheight, self.owidth)

        # ignore far away mappoints
        # sparse_np[sparse_np > self.max_depth ] = 0

        empty_depth = np.zeros((self.oheight,self.owidth))
        rgbd = np.append(rgb_np, np.expand_dims(empty_depth, axis=2), axis=2)
        
        # rgbd = np.append(rgb_np, np.expand_dims(sparse_np, axis=2), axis=2)

        input_tensor = to_tensor(rgbd)
        # 4, to emualte batch size 1
        while input_tensor.dim() < 4:
            input_tensor = input_tensor.unsqueeze(0)
 
        input_tensor = input_tensor.cuda()
        input_var = torch.autograd.Variable(input_tensor)

        depth_pred = self.model(input_var)
        depth_pred_np = np.squeeze(depth_pred.data.cpu().numpy())

        # if self.use_tello:
        #     focal_scale = 922./525
        #     depth_pred_np *= focal_scale

        depth_valid = sparse_np > 0
        depth_valid = np.bitwise_and(depth_valid, depth_pred_np > 0)

        if target_msg is not None:
            target_np = val_transform(target_msg, self.oheight, self.owidth)
            depth_valid = np.bitwise_and(depth_valid, target_np > 0)
            self.targ_points = np.append(self.targ_points, target_np[depth_valid])

        self.sparse_points = np.append(self.sparse_points, sparse_np[depth_valid])
        self.est_points = np.append(self.est_points, depth_pred_np[depth_valid])
        self.n_points = np.append(self.n_points, int(np.sum(depth_valid)))
        self.indexes = np.append(self.indexes, np.nonzero(depth_valid))
        
        _scales = self.est_points/self.sparse_points
        median_mask = median_filter(_scales, m=0.5)

        solve = minimize_scalar(self._scale_error, args=(self.sparse_points[~median_mask], self.est_points[~median_mask]))
        self.scale_est = solve.x

        if target_msg is not None:
            _scales = self.targ_points/self.sparse_points
            median_mask = median_filter(_scales, m=0.5)
            target_scale_min = minimize_scalar(self._scale_error, args=(self.sparse_points[~median_mask], self.targ_points[~median_mask]))
            print("est scale: %.3f, kinect scale : %.3f " % (self.scale_est, target_scale_min.x))
        else:
            print("scale estimate: %.3f " % self.scale_est)

    def predict_depth(self, rgb_msg, sparse_msg, target_msg=None):
        start_time = rospy.Time.now()
        rgb_np = val_transform(rgb_msg, self.oheight, self.owidth)
        sparse_np = val_transform(sparse_msg, self.oheight, self.owidth)

        # filter sparse outliers
        # rem_mask = statistical_outlier_removal(sparse_np, k=20, std_mul=1.0)
        # sparse_np[rem_mask] = 0
 
        if self.scale_est is not None:
            sparse_np = sparse_np*self.scale_est

        # ignore far away mappoints
        sparse_np[sparse_np > self.max_depth ] = 0

        # use amount of samples the network was trained for
        # n_samples = 200 
        # ni,nj = np.nonzero(sparse_np)
        # if len(ni) > n_samples:
        #     ri = np.random.choice(len(ni), n_samples, replace=False)
        #     inv_mask = np.ones((self.oheight, self.owidth), dtype=np.bool)
        #     inv_mask[ni[ri], nj[ri]] = 0
        #     sparse_np[inv_mask] = 0.0
 
        #     assert len(sparse_np[sparse_np>0.0]) == n_samples, "Not exactly n_points!"
        # elif len(ni) < 5:
        #     rospy.logwarn("Less then 5 points in sparse depth map: %d, returning nothing" % len(ni))
        #     return np.zeros_like(sparse_np), sparse_np
        # else: 
        #     rospy.logwarn("sparse depth map has less then %d points: %d" % (n_samples, len(ni)))
 
        # sparse_np = np.zeros((self.oheight,self.owidth), dtype=np.float32)

        rgbd = np.append(rgb_np, np.expand_dims(sparse_np, axis=2), axis=2)

        input_tensor = to_tensor(rgbd)
        # 4, to emualte batch size 1
        while input_tensor.dim() < 4:
            input_tensor = input_tensor.unsqueeze(0)
 
        input_tensor = input_tensor.cuda()
        input_var = torch.autograd.Variable(input_tensor)

        depth_pred = self.model(input_var)

        # add preknown points to prediction
        in_depth = input_tensor[:, 3:, :, :]
        in_valid = in_depth > 0
        depth_pred[in_valid] = in_depth[in_valid]

        depth_pred_np = np.squeeze(depth_pred.data.cpu().numpy())

        # remove points too far away
        depth_pred_np[depth_pred_np > self.max_depth] = np.nan



        if self.filter_low_texture:
            # kdtree outlier removal
            stat_mask = statistical_outlier_removal(depth_pred_np, k=40, std_mul=0.5) 
            depth_pred_np[stat_mask] = np.nan

            rmask= region_mask(sparse_np)
            cmask = convex_mask(rmask)
            depth_pred_np[~cmask] = np.nan

        if target_msg is not None:
            data_time = (rospy.Time.now()-start_time).to_sec()
            target_np = val_transform(target_msg, self.oheight, self.owidth)
            target_np[target_np > self.max_depth] = np.nan
            target_np[~(depth_pred_np > 0)] = np.nan
            target_tensor = to_tensor(target_np)
            target_tensor = target_tensor.unsqueeze(0)
            self.evaluate_results(depth_pred, target_tensor, data_time)
            self.sample_metrics_tracker.evaluate(sparse_np, target_np)

        # if self.use_tello:
        #     focal_scale = 922./525
        #     depth_pred_np *= focal_scale
        #     sparse_np *= focal_scale

        self.est_frame_saver.save_image(depth_pred_np, rgb_msg.header.stamp)
        # self.est_frame_saver.save_image(target_np, rgb_msg.header.stamp)
        self.rgb_frame_saver.save_image(rgb_np, rgb_msg.header.stamp)

        return depth_pred_np, sparse_np 

    def emulate_predict_depth(self, rgb_msg, target_msg):
        rgb_np = val_transform(rgb_msg, self.oheight, self.owidth)
        target_np = val_transform(target_msg, self.oheight, self.owidth)
        input_np = self.create_rgbd(rgb_np, target_np)

        input_tensor = to_tensor(input_np)
        # 4, to emualte batch size 1
        while input_tensor.dim() < 4:
            input_tensor = input_tensor.unsqueeze(0)
 
        target_tensor = to_tensor(target_np)
        target_tensor = target_tensor.unsqueeze(0)

        input_tensor, target_tensor = input_tensor.cuda(), target_tensor.cuda()
        input_var = torch.autograd.Variable(input_tensor)

        depth_pred = self.model(input_var)

        self.evaluate_results(depth_pred, target_tensor)

        # add preknown points to prediction
        in_depth = input_tensor[:, 3:, :, :]
        # in_valid = in_depth > 0.0
        # depth_pred[in_valid] = in_depth[in_valid]

        in_depth_np = np.squeeze(in_depth.data.cpu().numpy())
        depth_pred_np = np.squeeze(depth_pred.data.cpu().numpy())

        self.sample_metrics_tracker.evaluate(in_depth_np, target_np)

        self.est_frame_saver.save_image(target_np, rgb_msg.header.stamp)
        self.rgb_frame_saver.save_image(rgb_np, rgb_msg.header.stamp)

        return depth_pred_np, in_depth_np

    def evaluate_results(self, depth_pred, target_tensor, time=0, sparse=None):
        result = Result()
        output = torch.index_select(depth_pred.data, 1, torch.cuda.LongTensor([0]))
        target_tensor = target_tensor.unsqueeze(0)
        result.evaluate(output, target_tensor)
        self.average_meter.update(result, 0, time)
        avg_msg = get_result_msg(self.average_meter.average(), self.average_meter.count)
        self.avg_res_pub.publish(avg_msg)

    def run(self):
        rospy.loginfo("Started %s node" % rospy.get_name())
        rospy.spin()
        self.est_frame_saver.close()
        self.rgb_frame_saver.close()

        # TMP
        np.save("sparse_points.npy", self.sparse_points)
        np.save("est_points.npy", self.est_points)
        np.save("targ_points.npy", self.targ_points)
        np.save("n_depths.npy", self.n_points)
        np.save("indexes.npy", self.indexes)


        return
