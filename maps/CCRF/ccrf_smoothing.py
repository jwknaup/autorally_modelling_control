import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import qpSmooth


def reorder_points(pts):
    N = len(pts)
    old_pts = pts.copy()
    new_pts = np.zeros((N, 2))
    new_pts[0, 0] = old_pts[0, 0]
    new_pts[0, 1] = old_pts[0, 1]
    old_pts = np.delete(old_pts, 0, axis=0)
    for ii in range(1, N):
        closest = np.argmin(np.linalg.norm(new_pts[ii-1, :] - old_pts, axis=1))
        # print(new_pts[ii-1, :], old_pts[closest, :])
        # plt.plot(new_pts[:, 0], new_pts[:, 1])
        # plt.show()
        new_pts[ii, :] = old_pts[closest, :]
        old_pts = np.delete(old_pts, closest, axis=0)
    return new_pts


def generate_init_trajectory(img):
    thresh = 5
    pts = np.where(img[:, :, 3] < thresh)
    plt.plot(pts[1], pts[0])
    plt.imshow(img, cmap='gray')
    plt.show()
    pts = reorder_points(np.hstack((pts[1].reshape((-1, 1)), pts[0].reshape((-1, 1)))))
    # plt.plot(pts[:, 0], pts[:, 1])
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return pts


class MyQpSmooth(qpSmooth.QpSmooth):

    def __init__(self, init_pts, inner, outer, img):
        super().__init__(init_pts)
        self.img = img
        self.inner = inner
        self.outer = outer

    def checkTrackBoundary(self, coord, n, delta_max):
        # print(coord, n)
        delta_max = 2
        thresh = 1
        res = 0.01
        angle = np.arctan2(n[1], n[0])
        coord = np.array([coord[0], coord[1]])
        n = np.array([n[0], n[1]])
        F, R = 0, 0
        for F in np.arange(start=0, stop=delta_max, step=res):
            pt = (coord + (F) * n).reshape((2, 1))
            # print(pt.shape)
            dist_inner = np.min(np.linalg.norm(pt - self.inner, axis=0))
            dist_outer = np.min(np.linalg.norm(pt - self.outer, axis=0))
            dist = min(dist_inner, dist_outer)
            if dist < thresh:
                break
        for R in np.arange(start=0, stop=delta_max, step=res):
            pt = (coord - (R) * n).reshape((2, 1))
            # print(np.linalg.norm(pt - self.outer).shape)
            dist_inner = np.min(np.linalg.norm(pt - self.inner, axis=0))
            dist_outer = np.min(np.linalg.norm(pt - self.outer, axis=0))
            dist = min(dist_inner, dist_outer)
            if dist < thresh:
                break
        # print(F, R)
        return F, R

    def drawRaceline(self, lineColor=(0,0,255), img=None):
        self.resolution = 100

        u_new = np.linspace(0,len(self.break_pts),1000)
        xy = self.raceline_fun(u_new).reshape(-1,2)
        # print(xy.shape)
        x_new = xy[:,0]
        y_new = xy[:,1]
        plt.plot(x_new, y_new, 'c')
        plt.plot(self.inner[0, :], self.inner[1, :], 'k')
        plt.plot(self.outer[0, :], self.outer[1, :], 'k')
        plt.show()
        plt.plot(self.break_pts[:, 0], self.break_pts[:, 1])
        plt.plot(self.inner[0, :], self.inner[1, :], 'k')
        plt.plot(self.outer[0, :], self.outer[1, :], 'k')
        plt.show()


if __name__ == '__main__':
    file_name = 'CCRF_2021-01-10.npz'
    track_dict = np.load(file_name)
    centerline = np.vstack([track_dict['X_cen_smooth'], track_dict['Y_cen_smooth']])
    inner = np.vstack((track_dict['X_in'], track_dict['Y_in']))
    outer = np.vstack((track_dict['X_out'], track_dict['Y_out']))

    pts = centerline.T
    qp = MyQpSmooth(pts, inner, outer, None)
    val = qp.optimizePath()
    K, C, Ds = qp.curvatureJac()
    print(qp.break_pts.shape)
    print(K.shape)
    # print(K)
    # print(qp.break_pts)
    dif_vecs = qp.break_pts[1:, :] - qp.break_pts[:-1, :]
    s = np.cumsum(np.linalg.norm(dif_vecs, axis=1))
    print(s.shape)
    for ii in range(len(s)):
        print(s[ii], K[ii])
    plt.scatter(qp.break_pts[:, 0], qp.break_pts[:, 1], c=K, marker='.')
    plt.plot(inner[0, :], inner[1, :], 'k')
    plt.plot(outer[0, :], outer[1, :], 'k')
    plt.show()
    np.savez('ccrf_track_curvature.npz', pts=qp.break_pts, curvature=K, s=s)
