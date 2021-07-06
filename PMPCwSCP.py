import numpy as np
from scipy import sparse
import scipy.io
from matplotlib import pyplot as plt
import osqp
import cvxpy as cp
from mosek import fusion
import cs_model


class Controller:

    def __init__(self, n, m, N, M):
        self.n = n
        self.m = m
        self.N = N
        self.M = M
        self.l = n

        self.w_mean = 0
        self.w_stdev = 0.05

        self.model = cs_model.Model(N*M)
        self.solver = cvxpy_solver()

    def roll_out_trajectory(self, x_0, us, ws, lin=None):
        if lin:
            A, B, d = lin
            xs = np.dot(A, x_0) + np.dot(B, us.flatten().reshape((-1, 1))) + d
            xs = xs.reshape((self.n, self.N), order='F')
            xs = np.hstack((x_0, xs[:, :(N - 1)]))
        else:
            xs = np.zeros((self.n, self.N, self.M))
            xs[:, 0, :] = x_0.copy()
            for ii in range(N - 1):
                xs[:, ii + 1, :] = self.model.update_dynamics(xs[:, ii, :], us[:, ii, :], 0.1) + ws[:, ii, :]
            # print(x_0, us, xs)
        return xs

    def sample_x0(self, X0):
        return X0[0] + X0[1] * np.random.randn(self.n, self.M)

    def sample_w(self):
        w = self.w_mean + self.w_stdev * np.random.randn(self.l, self.N, self.M)
        w[3:, :, :] = 0
        return w

    def alg1(self, X0, xs, us, cartesian, dt):
        x0s = self.sample_x0(X0)
        ws = self.sample_w()
        xs[:, 0, :] = x0s
        xs = self.roll_out_trajectory(xs[:, 0, :], us, ws)
        xs, us = self.alg2(x0s, ws, xs, us)
        # print(xs, us)
        state, cartesian = self.model.update_dynamics(X0[0], us[:, 0:1, 0], dt, cartesian=cartesian)
        w = 0.05*np.random.randn(self.l, 1)
        w[3:, :] = 0
        state += w
        return (state, 0), xs, us, cartesian

    def alg2(self, x0s, ws, xs, us):
        eps = 5e-0
        while 1:
            xs = xs.reshape((self.n, -1))
            us = us.reshape((self.m, -1))
            A, B, d = self.model.linearize_dynamics(xs, us, dt=0.1)
            dfdxs = A.reshape((self.n, self.n, self.M, self.N), order='F')
            dfdus = B.reshape((self.n, self.m, self.M, self.N), order='F')
            dfdxs_blkdiag = np.zeros((self.n*self.M, self.n*self.M, self.N))
            temp = dfdxs[:, :, 0, 0]
            dfdus_blkdiag = np.zeros((self.n*self.M, self.m*self.M, self.N))
            for ii in range(M):
                dfdxs_blkdiag[ii*self.n:(ii+1)*self.n, ii*self.n:(ii+1)*self.n, :] = dfdxs[:, :,  ii, :]
                dfdus_blkdiag[ii*self.n:(ii+1)*self.n, ii*self.m:(ii+1)*self.m, :] = dfdus[:, :,  ii, :]
            temp = dfdxs_blkdiag[:, :, 0]
            xs = xs.reshape((self.n, self.N, self.M))
            us = us.reshape((self.m, self.N, self.M))
            xs = np.swapaxes(xs, 1, 2)
            us = np.swapaxes(us, 1, 2)
            xs = xs.reshape((self.n * self.M, self.N), order='F')
            us = us.reshape((self.m * self.M, self.N), order='F')
            dxs, dus = self.solver.solve_ocp_cvxpy(xs, us, dfdxs_blkdiag, dfdus_blkdiag)
            xs += dxs
            us += dus
            xs = xs.reshape((self.n, self.M, self.N), order='F')
            us = us.reshape((self.m, self.M, self.N), order='F')
            xs = np.swapaxes(xs, 1, 2)
            us = np.swapaxes(us, 1, 2)
            change = np.sum(np.linalg.norm(dxs, axis=0) + np.linalg.norm(dus, axis=0)) / self.N / self.M
            print(change)
            if change < eps:
                break
        return xs, us


class cvxpy_solver:
    def __init__(self):
        nx = 8
        nu = 2
        N = 10
        M = 5
        px = 0.0001
        pu = 0.0001
        Nc = 1

        Q = sparse.diags([3., 1., 1., 0., 0., 10., 10., 0.])
        # Q = sparse.diags([1., 1.])
        Q_bar = sparse.kron(sparse.eye(M), Q)
        QN = Q
        R = 0.1 * sparse.eye(2)
        R_bar = sparse.kron(sparse.eye(M), R)
        xr = np.tile(np.array([6, 0, 0, 0, 0, 0, 0, 0]), (M))

        dx = cp.Variable((nx * M, N))
        du = cp.Variable((nu * M, N))

        xs = cp.Parameter((nx * M, N))
        us = cp.Parameter((nu * M, N))
        self.dfdxs = cp.Parameter((nx * M, nx * M * N))
        self.dfdus = cp.Parameter((nx * M, nu * M * N))

        objective = 0
        for kk in range(N):
            objective += cp.quad_form(xs[:, kk] + dx[:, kk] - xr, Q_bar)
            objective += cp.quad_form(us[:, kk] + du[:, kk], R_bar)
        objective += px * cp.sum_squares(dx) + pu * cp.sum_squares(du)

        constraints = []
        constraints += [dx[:, 0] == 0]
        for kk in range(N - 1):
            constraints += [dx[:, kk + 1] == self.dfdxs[:, (nx * M)*kk:(nx * M)*(kk+1)] @ dx[:, kk] + self.dfdus[:, (nu * M)*kk:(nu * M)*(kk+1)] @ du[:, kk]]
        umin = -1 * np.ones((nu * M, 1))
        umax = np.ones((nu * M, 1))
        for kk in range(N):
            constraints += [umin <= us[:, kk:kk + 1] + du[:, kk:kk + 1], us[:, kk:kk + 1] + du[:, kk:kk + 1] <= umax]
            constraints += [-1 <= xs[6, kk] + dx[6, kk], xs[6, kk] + dx[6, kk] <= 1]
        for kk in range(Nc):
            constraints += [du[:-nu, kk] == du[nu:, kk]]

        self.ocp = cp.Problem(cp.Minimize(objective), constraints)
        self.dx = dx
        self.du = du
        self.xs = xs
        self.us = us
        self.nx = nx
        self.nu = nu
        self.M = M
        self.N = N

    def solve_ocp_cvxpy(self, xs, us, dfdxs, dfdus):
        # nx = 8
        # nu = 2
        # N = 10
        # M = 5
        # px = 0.001
        # pu = 0.001
        # Nc = 1
        #
        # Q = sparse.diags([3., 1., 1., 0., 0., 10., 10., 0.])
        # # Q = sparse.diags([1., 1.])
        # Q_bar = sparse.kron(sparse.eye(M), Q)
        # QN = Q
        # R = 0.1 * sparse.eye(2)
        # R_bar = sparse.kron(sparse.eye(M), R)
        # xr = np.tile(np.array([5, 0, 0, 0, 0, 0, 0, 0]), (M))
        #
        # dx = cp.Variable((nx*M, N))
        # du = cp.Variable((nu*M, N))
        #
        # objective = 0
        # for kk in range(N):
        #     objective += cp.quad_form(xs[:, kk] + dx[:, kk] - xr, Q_bar)
        #     objective += cp.quad_form(us[:, kk] + du[:, kk], R_bar)
        # objective += px * cp.sum_squares(dx) + pu * cp.sum_squares(du)
        #
        # constraints = []
        # constraints += [dx[:, 0] == 0]
        # for kk in range(N-1):
        #     constraints += [dx[:, kk+1] == dfdxs[:, :, kk] @ dx[:, kk] + dfdus[:, :, kk] @ du[:, kk]]
        # umin = -1 * np.ones((nu*M, 1))
        # umax = np.ones((nu*M, 1))
        # for kk in range(N):
        #     constraints += [umin <= us[:, kk:kk+1] + du[:, kk:kk+1], us[:, kk:kk+1] + du[:, kk:kk+1] <= umax]
        #     constraints += [-1 <= xs[6, kk] + dx[6, kk], xs[6, kk] + dx[6, kk] <= 1]
        # for kk in range(Nc):
        #     constraints += [du[:-nu, kk] == du[nu:, kk]]
        #
        # ocp = cp.Problem(cp.Minimize(objective), constraints)
        self.xs.value = xs
        self.us.value = us
        self.dfdxs.value = dfdxs.reshape((self.nx * self.M, self.nx * self.M * self.N), order='F')
        self.dfdus.value = dfdus.reshape((self.nx * self.M, self.nu * self.M * self.N), order='F')

        self.ocp.solve(solver=cp.OSQP, warm_start=True)
        return self.dx.value, self.du.value


# class OCP:
#
#     def __init__(self):
#         self.n = 2
#         self.m = 2
#         self.N = 10
#         self.M = 5
#
#         self.ocp = fusion.Model()
#
#     def define_ocp_mosek(self, f_bars, dfdxs, dfdus):
#         dx = self.ocp.variable((self.n, self.N, self.M), fusion.Domain.unbounded())
#         du = self.ocp.variable((self.m, self.N, self.M), fusion.Domain.unbounded())
#         Q_cost = self.ocp.variable((self.N, self.M), fusion.Domain.unbounded())
#
#         self.ocp.objective(fusion.ObjectiveSense.Minimize, fusion.Expr.add([Q_cost, R_cost, Expr.mul(px, dx_cost), Expr.mul(pu, du_cost)]))
#         for kk in range(N):
#             self.ocp.constraint(fusion.Expr.vstack(0.5, Q_cost.index(kk, ii), fusion.Expr.mul(Q_bar_half, fusion.Expr.add(xs + dx))),
#                                 fusion.Domain.inRotatedQCone())
#             self.ocp.constraint(fusion.Expr.vstack(0.5, R_cost, fusion.Expr.mul(R_bar_half, fusion.Expr.add(us + du))), fusion.Domain.inRotatedQCone())


def plot(states, controls, sim_length):
    plt.figure()
    time = np.arange(sim_length) / 10
    mat = scipy.io.loadmat("mppi_data/track_boundaries.mat")
    inner = mat['track_inner'].T
    outer = mat['track_outer'].T
    for ii in range(14):
        plt.subplot(7, 2, ii + 1)
        if ii < 11:
            plt.plot(time, states[ii, :, 0])
        elif ii == 11:
            plt.plot(states[-2, :, 0], states[-1, :, 0])
            plt.plot(inner[0, :], inner[1, :], 'k')
            plt.plot(outer[0, :], outer[1, :], 'k')
        else:
            plt.plot(time, controls[ii-12, :, 0])
    # states = np.load('cs_2Hz_states.npz.npy')
    # controls = np.load('cs_2Hz_control.npz.npy')
    # for ii in range(14):
    #     plt.subplot(7, 2, ii + 1)
    #     if ii < 11:
    #         plt.plot(time, states[ii, :])
    #     elif ii == 11:
    #         plt.plot(states[-2, :], states[-1, :])
    #         plt.plot(inner[0, :], inner[1, :], 'k')
    #         plt.plot(outer[0, :], outer[1, :], 'k')
    #     else:
    #         plt.plot(time, controls[ii-12, :])
    # add_labels()
    # mat = scipy.io.loadmat('mppi_data/track_boundaries.mat')
    # inner = mat['track_inner'].T
    # outer = mat['track_outer'].T
    # plt.subplot(7, 2, 12)
    # plt.plot(inner[0, :], inner[1, :], 'k')
    # plt.plot(outer[0, :], outer[1, :], 'k')
    # np.save('cs_5Hz_states', states)
    # np.save('cs_5Hz_control', controls)
    # plt.show()
    plt.figure()
    for ii in range(states.shape[2]):
        plt.plot(states[-2, :], states[-1, :], ii)
    # states = np.load('cs_5Hz_states.npy')
    # plt.plot(states[-2, :], states[-1, :])
    plt.plot(inner[0, :], inner[1, :], 'k')
    plt.plot(outer[0, :], outer[1, :], 'k')
    # plt.gca().legend(('ltv mpc', 'cs smpc', 'track boundaries'))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


if __name__ == '__main__':
    n = 8
    m = 2
    N = 10
    M = 5
    runs = 5
    sim_len = 200
    controller = Controller(n, m, N, M)
    sim_states = np.zeros((n+3, sim_len, runs))
    sim_controls = np.zeros((m, sim_len, runs))
    crashes = 0
    for jj in range(runs):
        state = np.array([4, 0., 0., 50, 50, 0.1, 0.0, 0]).reshape((-1, 1, 1))
        cartesian = np.array([-0.6613 + 0.1, 2.78 - 3.25, -2.97 + 2.3]).reshape((-1, 1, 1))
        control = np.array([0, 0.3]).reshape((-1, 1, 1))
        xs = np.tile(state, (1, N, M))
        us = np.tile(control, (1, N, M))
        state = state[:, :, 0]
        for ii in range(sim_len):
            # try:
            X0, xs, us, cartesian = controller.alg1((state, 0), xs, us, cartesian, 0.05)
            # except:
            #     crashes += 1
            #     break
            state = X0[0]
            if (np.abs(state[6, :]) > 1).any():
                crashes += 1
                break
            # xs[:, :-1] = xs[:, 1:]
            # xs[:, -1] = xs[:, -2]
            # us[:, :-1, :] = us[:, 1:, :]
            # us[:, -1, :] = us[:, -2, :]
            # xs[:, 0] = state.flatten()
            # xs = controller.roll_out_trajectory(xs[:, 0], us, N)
            sim_states[:n, ii, jj] = state.flatten()
            sim_states[n:, ii, jj] = cartesian.flatten()
            sim_controls[:, ii, jj] = us[:, 0, 0]
    # print(xs, us)
    print(crashes)
    plot(sim_states, sim_controls, sim_len)
