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
    pts = np.where(img < thresh)
    # plt.plot(pts[1], pts[0])
    # plt.imshow(img, cmap='gray')
    # plt.show()
    pts = reorder_points(np.hstack((pts[1].reshape((-1, 1)), pts[0].reshape((-1, 1)))))
    # plt.plot(pts[:, 0], pts[:, 1])
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return pts


class MyQpSmooth(qpSmooth.QpSmooth):

    def __init__(self, init_pts, img):
        super().__init__(init_pts)
        self.img = img

    def checkTrackBoundary(self, coord, n, delta_max):
        delta_max = 50
        thresh = 100
        buffer = 2
        angle = np.arctan2(n[1], n[0])
        coord = np.array([coord[0], coord[1]])
        n = np.array([n[0], n[1]])
        for F in range(delta_max):
            pt = (coord + (F + buffer) * n).astype(int)
            if self.img[pt[1], pt[0]] > thresh:
                break
        for R in range(delta_max):
            pt = (coord - (R + buffer) * n).astype(int)
            if self.img[pt[1], pt[0]] > thresh:
                break
        return F, R

    def drawRaceline(self, lineColor=(0,0,255), img=None):
        self.resolution = 100

        u_new = np.linspace(0,len(self.break_pts),1000)
        xy = self.raceline_fun(u_new).reshape(-1,2)
        # print(xy.shape)
        x_new = xy[:,0]
        y_new = xy[:,1]
        plt.imshow(self.img, cmap='gray')
        plt.plot(x_new, y_new, 'c')
        plt.show()
        plt.imshow(self.img, cmap='gray')
        plt.plot(self.break_pts[:, 0], self.break_pts[:, 1])
        plt.show()


if __name__ == '__main__':
    filename = 'mining_loop_lores.pgm'

    im = Image.open(filename)
    img = np.asarray(im)
    # im.show()
    # print(im)

    pts = generate_init_trajectory(img)
    qp = MyQpSmooth(pts, img)
    val = qp.optimizePath()
    K, C, Ds = qp.curvatureJac()
    print(qp.break_pts.shape)
    print(K.shape)
    # print(K)
    # print(qp.break_pts)
    dif_vecs = qp.break_pts[1:, :] - qp.break_pts[:-1, :]
    s = np.cumsum(np.linalg.norm(dif_vecs, axis=1))
    print(s.shape)
    np.savez('mining_loop_lores.npz', pts=qp.break_pts, curvature=K, s=s)
