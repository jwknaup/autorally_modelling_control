from mosek.fusion import *
import numpy as np
import scipy.linalg
import scipy.stats
import time
import cs_model
import matplotlib.pyplot as plt
import scipy.sparse


class PathSolver:
    def __init__(self, n, m, l, N):
        M = Model()
        self.n = n
        self.m = m
        self.l = l
        self.N = N

        V = M.variable("V", [m*N, 1], Domain.inRange(-1, 1))
        z = M.variable("z", [4, N], Domain.integral(Domain.inRange(0, 1)))
        Z = M.variable([n, n], Domain.unbounded())

        # linear variables with quadratic cone constraints
        w = M.variable("w", 1, Domain.unbounded())
        q = M.variable("q", 1, Domain.unbounded())

        mu_0_T_A_T_Q_bar_B = M.parameter([1, m*N])
        vec_T_sigma_y_Q_bar_B = M.parameter([1, n*N*m*N])
        pattern = []
        for ii in range(n*N):
            for jj in range(m*N):
                if jj <= ii:
                    pattern.append([ii, jj])
        pattern = []
        for ii in range(N):
            for jj in range(n):
                for kk in range(m * (ii + 1)):
                    pattern.append([jj + n * ii, kk])
        Q_bar_half_B = M.parameter([n*N, m*N], pattern)
        pattern = []
        for ii in range(m*N):
            pattern.append([ii, ii])
        R_bar_half = M.parameter([m*N, m*N], pattern)
        pattern = []
        for ii in range(n * N):
            pattern.append([ii, ii])
        Q_bar_half = M.parameter([n*N, n*N], pattern)
        # sigma_y_half = M.parameter([n*N, n*N])
        A_sigma_0_half = M.parameter([n*N, n])
        pattern = []
        for ii in range(N):
            for jj in range(n):
                for kk in range(l*(ii+1)):
                    pattern.append([jj + n * ii, kk])
        D = M.parameter([n*N, l*N])
        A_mu_0 = M.parameter([n*N, 1])
        pattern = []
        for ii in range(N):
            for jj in range(n):
                for kk in range(m * (ii + 1)):
                    pattern.append([jj + n * ii, kk])
        B = M.parameter([n*N, m*N])
        d = M.parameter([n*N, 1])
        sigma_N_inv = M.parameter([n, n])
        neg_x_0_T_Q_B = M.parameter([1, m*N])
        d_T_Q_B = M.parameter([1, m*N])
        u_0 = M.parameter([m, 1])

        I = Expr.constTerm(np.eye(n*N))

        M.constraint(Expr.sum(z.asExpr(), 0), Domain.equalsTo(3))
        # alpha = Matrix.sparse(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]))
        x_min = -2
        x_max = -1
        y_min = -1
        y_max = 1
        X = Expr.add(A_mu_0, Expr.mul(B, V))
        x_matrix = np.zeros((N, n*N))
        for ii in range(N):
            x_matrix[ii, ii*n] = 1
        y_matrix = np.zeros((N, n * N))
        for ii in range(N):
            y_matrix[ii, ii * n + 1] = 1
        xs = Expr.mul(Matrix.sparse(x_matrix), X)
        ys = Expr.mul(Matrix.sparse(y_matrix), X)
        gamma = 100
        print(Expr.flatten(z.slice([0, 0], [1, N])).getShape(), xs.getShape())
        M.constraint(Expr.sub(Expr.add(Expr.mul(Expr.flatten(z.slice([0, 0], [1, N])), -gamma), Expr.flatten(xs)), x_min),
                     Domain.lessThan(0))
        M.constraint(Expr.sub(Expr.add(Expr.mul(Expr.flatten(z.slice([1, 0], [2, N])), gamma), Expr.flatten(xs)), x_max),
                     Domain.greaterThan(0))
        M.constraint(Expr.sub(Expr.add(Expr.mul(Expr.flatten(z.slice([2, 0], [3, N])), -gamma), Expr.flatten(ys)), y_min),
                     Domain.lessThan(0))
        M.constraint(Expr.sub(Expr.add(Expr.mul(Expr.flatten(z.slice([3, 0], [4, N])), gamma), Expr.flatten(ys)), y_max),
                     Domain.greaterThan(0))

        # convert to linear objective with quadratic cone constraints
        M.objective(ObjectiveSense.Minimize, Expr.add([w, q]))
        M.constraint(Expr.vstack(0.5, w, Expr.flatten(Expr.mul(Q_bar_half, X))),
                     Domain.inRotatedQCone())
        M.constraint(Expr.vstack(0.5, q, Expr.flatten(Expr.mul(R_bar_half, V))),
                     Domain.inRotatedQCone())

        # M.constraint(Z, Domain.inPSDCone())

        # M.setSolverParam('mioTolAbsGap', 1000000000)

        self.M = M
        self.V = V
        self.z=z
        self.mu_0_T_A_T_Q_bar_B = mu_0_T_A_T_Q_bar_B
        self.vec_T_sigma_y_Q_bar_B = vec_T_sigma_y_Q_bar_B
        self.Q_bar_half_B = Q_bar_half_B
        self.R_bar_half = R_bar_half
        self.Q_bar_half = Q_bar_half
        # self.sigma_y_half = sigma_y_half
        self.A_sigma_0_half = A_sigma_0_half
        self.D = D
        self.A_mu_0 = A_mu_0
        self.B = B
        self.d = d
        self.sigma_N_inv = sigma_N_inv
        self.neg_x_0_T_Q_B = neg_x_0_T_Q_B
        self.d_T_Q_B = d_T_Q_B
        self.u_0 = u_0

    def populate_params(self, A, B, d, D, mu_0, sigma_0, sigma_N_inv, Q_bar, R_bar, u_0, x_target, K=None):
        n = 8
        m = 2
        l = 8
        N = self.N

        # A = np.tile(np.eye(n), (N, 1))
        # B = np.kron(np.eye(N), np.random.randn(n, m))
        # D = np.kron(np.eye(N), np.random.randn(n, l))
        # mu_0 = np.zeros((n, 1))
        # sigma_0 = np.random.randn(n, n)
        # sigma_0 = np.dot(sigma_0, sigma_0.T)
        # sigma_N_inv = np.linalg.inv(1000 * sigma_0)
        sigma_y = np.dot(A, np.dot(sigma_0, A.T)) + np.dot(D, D.T)
        # sigma_y = np.linalg.cholesky(sigma_y)
        # Q_bar = np.eye(n*N)
        # R_bar = np.eye(m*N)
        x_0 = x_target.copy()

        self.mu_0_T_A_T_Q_bar_B.setValue(2*np.dot(np.dot(np.dot(mu_0.T, A.T), Q_bar), B))
        temp = 2*np.dot(sigma_y, np.dot(Q_bar, B)).reshape((-1, 1)).T
        self.vec_T_sigma_y_Q_bar_B.setValue(temp)
        # try:
        #     self.Q_bar_half_B.setValue(np.dot(scipy.linalg.cholesky(Q_bar), B))
        # except np.linalg.LinAlgError:
        self.Q_bar_half_B.setValue(np.dot(np.sqrt(Q_bar), B))
        self.Q_bar_half.setValue(np.sqrt(Q_bar))
        # try:
        #     self.R_bar_half.setValue(scipy.linalg.cholesky(R_bar))
        # except np.linalg.LinAlgError:
        self.R_bar_half.setValue(np.sqrt(R_bar))
        # try:
        #     self.Q_bar_half.setValue(scipy.linalg.cholesky(Q_bar))
        # except np.linalg.LinAlgError:
        #     self.Q_bar_half.setValue(np.sqrt(Q_bar))
        # try:
        #     self.sigma_y_half.setValue(sigma_y)
        # except np.linalg.LinAlgError:
        #     print("cholesky failed")
        #     self.sigma_y_half.setValue(np.sqrt(sigma_y))
        # try:
        #     self.A_sigma_0_half.setValue(np.dot(A, np.linalg.cholesky(sigma_0)))
        # except np.linalg.LinAlgError:
        self.A_sigma_0_half.setValue(np.dot(A, np.sqrt(sigma_0)))
        self.D.setValue(D)
        self.A_mu_0.setValue((np.dot(A, mu_0)))
        self.B.setValue(B)
        self.d.setValue(d)
        # self.sigma_N_inv.setValue(sigma_N_inv)
        # self.neg_x_0_T_Q_B.setValue(2*np.dot(np.dot(-x_0.T, Q_bar), B))
        self.d_T_Q_B.setValue(2*np.dot(np.dot(d.T, Q_bar), B))

    def solve(self):
        # print(self.u_0.getValue())
        # self.M.solve()
        t0 = time.time()
        self.M.solve()
        print((time.time() - t0))
        try:
            K_level = self.V.level()
            levels = ( K_level, self.z.level())
            # print(levels)
            return levels
        except SolutionError:
            raise RuntimeError


def form_long_matrices_LTV(A, B, d, D, N):
    nx = A.shape[1]
    nu = B.shape[1]
    nl = D.shape[1]

    AA = np.zeros((nx*N, nx))
    BB = np.zeros((nx*N, nu * N))
    dd = np.zeros((nx*N, 1))
    DD = np.zeros((nx*N, nl * N))
    AA_i_row = np.eye(nx)
    dd_i_row = np.zeros((nx, 1))
    # B_i_row = zeros((nx, 0))
    # D_i_bar = zeros((nx, nx))
    for ii in np.arange(0, N):
        AA_i_row = np.dot(A[:, :, ii], AA_i_row)
        AA[ii*nx:(ii+1)*nx, :] = AA_i_row

        B_i_row = B[:, :, ii]
        D_i_row = D[:, :, ii]
        for jj in np.arange(ii-1, -1, -1):
            B_i_cell = np.dot(A[:, :, ii], BB[(ii-1)*nx:ii*nx, jj*nu:(jj+1)*nu])
            B_i_row = np.hstack((B_i_cell, B_i_row))
            D_i_cell = np.dot(A[:, :, ii], DD[(ii-1)*nx:ii*nx, jj*nl:(jj+1)*nl])
            D_i_row = np.hstack((D_i_cell, D_i_row))
        BB[ii*nx:(ii+1)*nx, :(ii+1)*nu] = B_i_row
        DD[ii*nx:(ii+1)*nx, :(ii+1)*nl] = D_i_row

        dd_i_row = np.dot(A[:, :, ii], dd_i_row) + d[:, :, ii]
        dd[ii*nx:(ii+1)*nx, :] = dd_i_row

    return AA, BB, dd, DD


if __name__ == '__main__':
    n = 2
    m = 2
    l = 1
    N = 50
    dt = 0.1
    sigma_w = 0.5
    A = np.array([[1, 0], [0, 1]])
    A1 = np.tile(A.reshape((n, n, 1)), (1, 1, N))
    B = np.array([[dt, 0], [0, dt]])
    B1 = np.tile(B.reshape((n, m, 1)), (1, 1, N))
    d = np.zeros((2, 1))
    d1 = np.tile(d.reshape((n, 1, 1)), (1, 1, N))
    D = np.zeros((n, l))
    D1 = np.tile(D.reshape((n, l, 1)), (1, 1, N))
    AA, BB, dd, DD = form_long_matrices_LTV(A1, B1, d1, D1, N)
    mu_0 = np.array([[-3], [0.1]])
    sigma_0 = np.zeros((2, 2))
    sigma_N = np.ones((2, 2))*100
    Q_bar = np.eye(N*n)
    R_bar = np.eye(N*m)

    spd_const = PathSolver(n, m, l, N)
    spd_const.populate_params(AA, BB, dd, DD, mu_0, sigma_0, sigma_N, Q_bar, R_bar, 0, np.zeros((n, 1)))
    file = np.load('myfile.npz')
    V1 = file['V']
    z1 = file['z']
    spd_const.V.setLevel(V1)
    spd_const.z.setLevel(z1)
    V, z = spd_const.solve()
    # np.savez('myfile', V=V, z=z)
    V = V.reshape((-1, 1))
    print(V, z.reshape((4, N)))
    X_bar = np.dot(AA, mu_0) + np.dot(BB, V)
    X = X_bar.reshape((n, N), order='F')
    plt.plot(X[0, :], X[1, :])
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)

    # time = np.arange(start=0, stop=dt*N, step=dt)
    # terminal_states = np.zeros((n, 100))
    # for jj in range(100):
    #     states = np.zeros((n, N))
    #     state = mu_0.copy()
    #     for ii in range(N):
    #         K_ii = K[ii * m: (ii + 1) * m, ii * n: (ii + 1) * n]
    #         if ii > 0:
    #             X_bar_ii = X_bar[(ii-1)*n:(ii)*n, :]
    #         else:
    #             X_bar_ii = mu_0.copy()
    #         y = state - X_bar_ii
    #         u = np.dot(K_ii, y)
    #         w = np.random.randn(1, 1)
    #         state = np.dot(A, state) + np.dot(B, u) + np.dot(D, w)
    #         states[:, ii] = state.flatten()
    #     plt.plot(states[0, :], states[1, :])
    #     terminal_states[:, jj] = states[:, -1]
    plt.show()
    # terminal_covariance = np.cov(terminal_states)
    # print(terminal_covariance)




