from mosek.fusion import *
import numpy as np
import scipy.linalg
import scipy.stats
import time
import cs_model
import matplotlib.pyplot as plt
import scipy.sparse


class CSSolver:
    def __init__(self, n, m, l, N, BB):
        M = Model()
        self.n = n
        self.m = m
        self.l = l
        self.N = N

        k = None
        pattern = []
        # for ii in range(N):
        #     for jj in range(m):
        #         for ll in range(n):
        #             row = ii*m + jj
        #             col = ii*n + ll
        #             pattern.append([row, col])
        for ii in range(m*N):
            for jj in range(n*N):
                row = ii
                col = jj
                pattern.append([row, col])
                if jj >= n*(ii+1):
                    break
        K = M.variable([m*N, n*N], Domain.sparse(Domain.unbounded(), pattern))

        # linear variables with quadratic cone constraints
        y1 = M.variable("y1", 1, Domain.unbounded())
        z1 = M.variable("z1", 1, Domain.unbounded())
        y2 = M.variable("y2", 1, Domain.unbounded())
        z2 = M.variable("z2", 1, Domain.unbounded())

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

        # convert to linear objective with quadratic cone constraints
        M.objective(ObjectiveSense.Minimize, Expr.add([y1, y2, z1, z2]))

        M.constraint(Expr.vstack(0.5, y1, Expr.flatten(Expr.mul(Q_bar_half, Expr.mul(Expr.add(I, Expr.mul(B, K)), A_sigma_0_half)))), Domain.inRotatedQCone())
        # check BKD multiplicaion, or maybe with I? Something seems wrong here b/c sparsity pattern is invalid
        M.constraint(Expr.vstack(0.5, y2, Expr.flatten(Expr.mul(Q_bar_half, Expr.mul(Expr.add(I, Expr.mul(B, K)), D)))), Domain.inRotatedQCone())
        M.constraint(Expr.vstack(0.5, z1, Expr.flatten(Expr.mul(R_bar_half, Expr.mul(K, A_sigma_0_half)))), Domain.inRotatedQCone())
        M.constraint(Expr.vstack(0.5, z2, Expr.flatten(Expr.mul(R_bar_half, Expr.mul(K, D)))), Domain.inRotatedQCone())

        # terminal covariance constraint
        sigma_N_inv = Expr.constTerm(np.ones((n, n))*.00001)
        # e_n = np.eye(n)
        # E_N = Matrix.sparse(np.hstack((np.zeros((n, (N - 1) * n)), e_n)))
        # Z = Expr.mul(E_N, Expr.mul(Expr.add(I, Expr.mul(B, K)), D))
        # # print(Expr.transpose(Z).getDim(0), I.getDim(0))
        # X_sym = Expr.vstack(Expr.hstack(sigma_N_inv, Z), Expr.hstack(Expr.transpose(Z), Expr.constTerm(np.eye(l*N))))
        # M.constraint(X_sym, Domain.inPSDCone())

        # S = 0.5*BB.copy()
        # EN = scipy.sparse.csr_matrix(np.hstack((np.zeros((n, (N-1) * n)), np.eye(n))))
        # BNbar = EN.dot(BB.copy())
        # Z = Expr.add(Matrix.sparse(EN.dot(S)), Expr.mul(Expr.mul(Matrix.sparse(BNbar), K), Matrix.dense(S)))
        # X_sym = Expr.vstack(Expr.hstack(sigma_N_inv, Z), Expr.hstack(Expr.transpose(Z), Expr.constTerm(np.eye(l * N))))
        # M.constraint(X_sym, Domain.inPSDCone())

        EN = np.hstack((np.zeros((n, (N - 1) * n)), np.eye(n)))
        # BNbar = EN.dot(BB.copy())
        # Z = Expr.add(Matrix.sparse(EN.dot(S)), Expr.mul(Expr.mul(Matrix.sparse(BNbar), K), Matrix.dense(S)))
        Z = Expr.add(Expr.mul(Expr.constTerm(EN), D), Expr.mul(EN, Expr.mul(B, Expr.mul(K, D))))
        X_sym = Expr.vstack(Expr.hstack(sigma_N_inv, Z), Expr.hstack(Expr.transpose(Z), Expr.constTerm(np.eye(l * N))))
        M.constraint(X_sym, Domain.inPSDCone())

        self.M = M
        self.k = k
        self.K = K
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
            K_level = self.K.level()
            levels = ( K_level)
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
    m = 1
    l = 1
    N = 10
    dt = 0.1
    sigma_w = 0.5
    A = np.array([[1, dt], [0, 1]])
    A1 = np.tile(A.reshape((n, n, 1)), (1, 1, N))
    B = np.array([[0], [dt]])
    B1 = np.tile(B.reshape((n, m, 1)), (1, 1, N))
    d = np.zeros((2, 1))
    d1 = np.tile(d.reshape((n, 1, 1)), (1, 1, N))
    D = sigma_w * B
    D1 = np.tile(D.reshape((n, l, 1)), (1, 1, N))
    AA, BB, dd, DD = form_long_matrices_LTV(A1, B1, d1, D1, N)
    mu_0 = np.array([[0], [1]])
    sigma_0 = np.zeros((2, 2))
    sigma_N = np.ones((2, 2))*100
    Q_bar = np.zeros((N*n, N*n))
    R_bar = np.eye(N*m)

    spd_const = CSSolver(n, m, l, N, BB)
    spd_const.populate_params(AA, BB, dd, DD, mu_0, sigma_0, sigma_N, Q_bar, R_bar, 0, np.zeros((n, 1)))
    K_flat = spd_const.solve()
    K = K_flat.reshape((m*N, n*N))
    print(K)
    X_bar = np.dot(AA, mu_0)

    time = np.arange(start=0, stop=dt*N, step=dt)
    terminal_states = np.zeros((n, 100))
    for jj in range(100):
        states = np.zeros((n, N))
        state = mu_0.copy()
        for ii in range(N):
            K_ii = K[ii * m: (ii + 1) * m, ii * n: (ii + 1) * n]
            if ii > 0:
                X_bar_ii = X_bar[(ii-1)*n:(ii)*n, :]
            else:
                X_bar_ii = mu_0.copy()
            y = state - X_bar_ii
            u = np.dot(K_ii, y)
            w = np.random.randn(1, 1)
            state = np.dot(A, state) + np.dot(B, u) + np.dot(D, w)
            states[:, ii] = state.flatten()
        plt.plot(states[0, :], states[1, :])
        terminal_states[:, jj] = states[:, -1]
    plt.show()
    terminal_covariance = np.cov(terminal_states)
    print(terminal_covariance)




