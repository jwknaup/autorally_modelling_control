import numpy as np
import matplotlib.pyplot as plt
import time
import rospy
from nav_msgs.msg import Odometry
from autorally_msgs.msg import wheelSpeeds
from scipy.spatial.transform import Rotation
from autorally_private_msgs.msg import mapCA


class MapCA:

    def __init__(self):
        file_name = 'Marietta_2021-01-09.npz'
        track_dict = np.load(file_name)
        p_x = track_dict['X_cen_smooth']
        p_y = track_dict['Y_cen_smooth']
        self.p = np.array([p_x, p_y])
        dif_vecs = self.p[:, 1:] - self.p[:, :-1]
        self.dif_vecs = dif_vecs
        self.slopes = dif_vecs[1, :] / dif_vecs[0, :]
        self.midpoints = self.p[:, :-1] + dif_vecs/2
        self.s = np.cumsum(np.linalg.norm(dif_vecs, axis=0))

        self.wf = 0.1
        self.wr = 0.1

        # plt.plot(p_x, p_y, '.-')
        # plt.plot(self.midpoints[0], self.midpoints[1], 'x')
        # plt.show()

        rospy.init_node('map_ca', anonymous=False)
        rospy.Subscriber("/pose_estimate", Odometry, self.odom_cb)
        rospy.Subscriber("/wheelSpeeds", wheelSpeeds, self.wheel_cb)
        self.mapca_pub = rospy.Publisher('/MAP_CA/mapCA', mapCA, queue_size=1)

    def localize(self, M, psi):
        dists = np.linalg.norm(np.subtract(M.reshape((-1,1)), self.midpoints), axis=0)
        mini = np.argmin(dists)
        p0 = self.p[:, mini]
        p1 = self.p[:, mini+1]
        # plt.plot(M[0], M[1], 'x')
        # plt.plot(p0[0], p0[1], 'o')
        # plt.plot(p1[0], p1[1], 'o')
        ortho = -1/self.slopes[mini]
        a = M[1] - ortho * M[0]
        a_0 = p0[1] - ortho*p0[0]
        a_1 = p1[1] - ortho*p1[0]
        printi=0
        if a_0 < a < a_1 or a_1 < a < a_0:
            norm_dist = np.sign(np.cross(p1 - p0, M - p0)) * np.linalg.norm(np.cross(p1 - p0, M - p0)) / np.linalg.norm(p1 - p0)
            s_dist = np.linalg.norm(np.dot(M-p0, p1-p0))
        else:
            printi=1
            norm_dist = np.sign(np.cross(p1 - p0, M - p0)) * np.linalg.norm(M - p0)
            s_dist = 0
        s_dist += self.s[mini]
        head_dist = psi - np.arctan2(self.dif_vecs[1, mini], self.dif_vecs[0, mini])
        if head_dist > np.pi:
            # print(psi, np.arctan2(self.dif_vecs[1, mini], self.dif_vecs[0, mini]))
            head_dist -= 2 * np.pi
        elif head_dist < -np.pi:
            head_dist += 2 * np.pi
        if abs(head_dist > np.pi):
            print(norm_dist, s_dist, head_dist * 180 / np.pi)
        # plt.show()
        return head_dist, norm_dist, s_dist

    def odom_cb(self, odom):
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        r = Rotation.from_quat([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])
        yaw, pitch, roll = r.as_euler('zyx', degrees=False)
        vx = odom.twist.twist.linear.x * np.cos(yaw) + odom.twist.twist.linear.y * np.sin(yaw)
        vy = odom.twist.twist.linear.x * -np.sin(yaw) + odom.twist.twist.linear.y * np.cos(yaw)
        wz = odom.twist.twist.angular.z

        d_psi, n, s = self.localize(np.asarray([x, y]), yaw)
        mapca = mapCA()
        mapca.header.stamp = rospy.Time.now()
        mapca.vx = vx
        mapca.vy = vy
        mapca.wz = wz
        mapca.wf = self.wf
        mapca.wr = self.wr
        mapca.s = s
        mapca.ey = n
        mapca.epsi = d_psi
        mapca.x = x
        mapca.y = y
        mapca.yaw = yaw
        mapca.path_s = s
        self.mapca_pub.publish(mapca)

    def wheel_cb(self, speeds):
        self.wf = (speeds.lfSpeed + speeds.rfSpeed) / 2.0
        self.wr = (speeds.lbSpeed + speeds.rbSpeed) / 2.0


if __name__ == '__main__':
    map_ca = MapCA()
    rospy.spin()
