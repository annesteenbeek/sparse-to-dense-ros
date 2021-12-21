import rospy
import numpy as np
from sparse_to_dense.msg import SampleMetrics


class SampleMetricsTracker(object):
    def __init__(self, max_depth=np.inf):
        self.metric_pub = rospy.Publisher('sample_metrics', SampleMetrics, queue_size=5)
        self.max_depth = max_depth
        self.samples = []
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_stde, self.sum_error = 0, 0
        self.count = 0

    def evaluate(self, sparse_np, target_np):

        depth_valid = sparse_np > 0
        depth_valid = np.bitwise_and(depth_valid, target_np > 0)
        if self.max_depth is not np.inf:
            depth_valid = np.bitwise_and(depth_valid, sparse_np <= self.max_depth)

        n_frame = np.count_nonzero(depth_valid)
        if n_frame <= 0:
            return

        self.count += 1
        self.samples.append(n_frame)
        error = target_np[depth_valid] - sparse_np[depth_valid]
        # error = error[~np.isnan(error)]
        abs_diff = np.absolute(error)
        self.sum_error += error.mean()
        self.sum_mae += abs_diff.mean()
        mse = (abs_diff**2).mean()
        self.sum_mse += mse
        self.sum_rmse += np.sqrt(mse)
        self.sum_stde += np.std(error)

        self.publish(n_frame)

    def publish(self, n_frame):
        msg = SampleMetrics()

        msg.max_depth = self.max_depth
        msg.n_frame = int(n_frame)
        msg.n_avg = np.mean(self.samples)
        msg.min = min(self.samples)
        msg.max = max(self.samples)
        msg.stddev = np.std(self.samples)
        msg.mse = self.sum_mse / self.count
        msg.rmse = self.sum_rmse / self.count
        msg.mae = self.sum_mae / self.count
        msg.me = self.sum_error / self.count
        msg.stde = self.sum_stde /self.count
        msg.count = self.count

        self.metric_pub.publish(msg)
