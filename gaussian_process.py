import time as tt
import numpy as np
from numpy import sin, cos, tan, arctan as atan, sqrt, arctan2 as atan2, zeros, zeros_like, abs, pi
import scipy.io
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import torch
from sklearn import gaussian_process
import imageio

import throttle_model
import cs_solver_obstacle

def add_labels(states, labels):
    rows = int(len(states) / 2)
    y_labels = ['$v_x$ m/s', '$v_y$ m/s', '$\omega_z$ rad/s', '$\omega_F$ rad/s', '$\omega_R$ rad/s', 'Yaw rad', 'X m', 'Y m']
    for ii, state_index in enumerate(states):
        plt.subplot(rows, 2, ii + 1)
        plt.gca().legend(labels)
        plt.xlabel('t (s)')
        plt.ylabel(y_labels[state_index])


class MapCA:

    def __init__(self):
        file_name = 'maps/CCRF/CCRF_2021-01-10.npz'
        track_dict = np.load(file_name)
        p_x = track_dict['X_cen_smooth']
        p_y = track_dict['Y_cen_smooth']
        file_name = 'maps/CCRF/ccrf_track_optimal.npz'
        track_dict = np.load(file_name)
        try:
            p_x = track_dict['X_cen_smooth']
            p_y = track_dict['Y_cen_smooth']
        except KeyError:
            p_x = track_dict['pts'][::-1, 0]
            p_y = track_dict['pts'][::-1, 1]
            self.rho = -track_dict['curvature'][::-1]
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
        while head_dist > np.pi:
            # print(psi, np.arctan2(self.dif_vecs[1, mini], self.dif_vecs[0, mini]))
            head_dist -= 2*np.pi
            print(norm_dist, s_dist, head_dist * 180 / np.pi)
        while head_dist < -np.pi:
            head_dist += 2*np.pi
            print(norm_dist, s_dist, head_dist * 180 / np.pi)
        # if printi:
        #     print(norm_dist, s_dist, head_dist*180/np.pi)
        #     printi=0
        # plt.show()
        return head_dist, norm_dist, s_dist


class Model:
    def __init__(self, N, cartesian=True):
        self.throttle = throttle_model.Net()
        self.throttle.load_state_dict(torch.load('throttle_model1.pth'))
        self.N = N
        self.cartesian = cartesian
        ccrf_curv = np.load('maps/CCRF/ccrf_track_optimal.npz')
        self.s = ccrf_curv['s']
        self.rho = -ccrf_curv['curvature'][::-1]

    def get_curvature(self, s):
        while (s > self.s[-1]).any():
            s[s > self.s[-1]] -= self.s[-1]

        dif = np.abs(s.reshape((-1, 1)) - self.s.reshape((1, -1)))
        idx = np.argmin(dif, axis=1)
        rho = self.rho[idx]
        return rho.flatten()

    def update_dynamics(self, state, input, dt, nn=None):
        state = state.T
        input = input.T
        m_Vehicle_m = 21.7562#1270
        m_Vehicle_Iz = 1.124#2000
        m_Vehicle_lF = 0.34#1.015
        lFR = 0.57#3.02
        m_Vehicle_lR = lFR-m_Vehicle_lF
        m_Vehicle_IwF = 0.1#8
        m_Vehicle_IwR = .0373
        m_Vehicle_rF = 0.095#0.325
        m_Vehicle_rR = 0.090#0.325
        m_Vehicle_h = 0.12#.54
        m_g = 9.80665

        tire_B = 4.0#10
        tire_C = 1.0
        tire_D = 1.0
        tire_E = 1.0
        tire_Sh = 0.0
        tire_Sv = 0.0

        N, dx = state.shape
        m_nu = 1

        vx = state[:, 0]
        vy = state[:, 1]
        wz = state[:, 2]
        wF = state[:, 3]
        wR = state[:, 4]
        psi = state[:, 5]
        X = state[:, 6]
        Y = state[:, 7]

        if (vx < 0.1).any():
            vx = np.maximum(vx, 0.1)
        if (wF < 1).any():
            wF = np.maximum(wF, 1)
        if (wR < 1).any():
            wR = np.maximum(wR, 1)

        m_Vehicle_kSteering = -0.24  # -pi / 180 * 18.7861
        m_Vehicle_cSteering = -0.02  # 0.0109
        # m_Vehicle_kSteering = 18.7861
        # m_Vehicle_cSteering = 0.0109
        throttle_factor = 0.38
        # delta = input[:, 0]
        steering = input[:, 0]
        delta = m_Vehicle_kSteering * steering + m_Vehicle_cSteering
        T = np.maximum(input[:, 1], 0)

        min_velo = 0.1
        deltaT = 0.01
        t = 0

        while t < dt:
            beta = atan2(vy, vx)

            V = sqrt(vx * vx + vy * vy)
            vFx = V * cos(beta - delta) + wz * m_Vehicle_lF * sin(delta)
            vFy = V * sin(beta - delta) + wz * m_Vehicle_lF * cos(delta)
            vRx = vx
            vRy = vy - wz * m_Vehicle_lR

            # sEF = -(vFx - wF * m_Vehicle_rF) / (vFx) + tire_Sh
            # muFx = tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv
            # sEF = -(vRx - wR * m_Vehicle_rR) / (vRx) + tire_Sh
            # muRx = tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv
            #
            # sEF = atan(vFy / abs(vFx)) + tire_Sh
            # alpha = -sEF
            # muFy = -tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv
            # sEF = atan(vRy / abs(vRx)) + tire_Sh
            # alphaR = -sEF
            # muRy = -tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv

            sFx = np.where(wF > 0, (vFx - wF * m_Vehicle_rF) / (wF * m_Vehicle_rF), 0)
            sRx = np.where(wR > 0, (vRx - wR * m_Vehicle_rR) / (wR * m_Vehicle_rR), 0)
            # sFy = np.where(vFx > 0, (1 + sFx) * vFy / vFx, 0)
            # sRy = np.where(vRx > 0, (1 + sRx) * vRy / vRx, 0)
            sFy = np.where(vFx > 0, vFy / (wF * m_Vehicle_rF), 0)
            sRy = np.where(vRx > 0, vRy / (wR * m_Vehicle_rR), 0)

            sF = np.sqrt(sFx * sFx + sFy * sFy) + 1e-2
            sR = np.sqrt(sRx * sRx + sRy * sRy) + 1e-2

            muF = tire_D * sin(tire_C * atan(tire_B * sF))
            muR = tire_D * sin(tire_C * atan(tire_B * sR))

            muFx = -sFx / sF * muF
            muFy = -sFy / sF * muF
            muRx = -sRx / sR * muR
            muRy = -sRy / sR * muR

            fFz = m_Vehicle_m * m_g * (m_Vehicle_lR - m_Vehicle_h * muRx) / (
                    m_Vehicle_lF + m_Vehicle_lR + m_Vehicle_h * (muFx * cos(delta) - muFy * sin(delta) - muRx))
            # fFz = m_Vehicle_m * m_g * (m_Vehicle_lR / 0.57)
            fRz = m_Vehicle_m * m_g - fFz

            fFx = fFz * muFx
            fRx = fRz * muRx
            fFy = fFz * muFy
            fRy = fRz * muRy

            dot_X = cos(psi)*vx - sin(psi)*vy
            dot_Y = sin(psi)*vx + cos(psi)*vy

            next_state = zeros_like(state)
            next_state[:, 0] = vx + deltaT * ((fFx * cos(delta) - fFy * sin(delta) + fRx) / m_Vehicle_m + vy * wz)
            next_state[:, 1] = vy + deltaT * ((fFx * sin(delta) + fFy * cos(delta) + fRy) / m_Vehicle_m - vx * wz)
            next_state[:, 2] = wz + deltaT * (
                        (fFy * cos(delta) + fFx * sin(delta)) * m_Vehicle_lF - fRy * m_Vehicle_lR) / m_Vehicle_Iz
            next_state[:, 3] = wF - deltaT * m_Vehicle_rF / m_Vehicle_IwF * fFx
            input_tensor = torch.from_numpy(np.hstack((T.reshape((-1, 1)), wR.reshape((-1, 1)) / throttle_factor))).float()
            next_state[:, 4] = wR + deltaT * self.throttle(input_tensor).detach().numpy().flatten()
            if self.cartesian:
                next_state[:, 5] = psi + deltaT * wz
                next_state[:, 6] = X + deltaT * dot_X
                next_state[:, 7] = Y + deltaT * dot_Y
            else:
                rho = self.get_curvature(Y)
                next_state[:, 5] = psi + deltaT * (wz - (vx * cos(psi) - vy * sin(psi)) / (1 - rho * X) * rho)
                next_state[:, 6] = X + deltaT * (vx * sin(psi) + vy * cos(psi))
                next_state[:, 7] = Y + deltaT * (vx * cos(psi) - vy * sin(psi)) / (1 - rho * X)

            t += deltaT
            vx = next_state[:, 0]
            vy = next_state[:, 1]
            wz = next_state[:, 2]
            wF = next_state[:, 3]
            wR = next_state[:, 4]
            psi = next_state[:, 5]
            X = next_state[:, 6]
            Y = next_state[:, 7]

        # print(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4)

        return next_state.T

    def linearize_dynamics(self, states, controls, dt=0.05):
        nx = 8
        nu = 2
        nN = states.shape[1]

        delta_x = np.array([0.01, 0.001, 0.01, 0.1, 0.1, 0.05, 0.1, 0.2])
        delta_u = np.array([0.01, 0.01])
        delta_x_flat = np.tile(delta_x, (1, nN))
        delta_u_flat = np.tile(delta_u, (1, nN))
        delta_x_final = np.multiply(np.tile(np.eye(nx), (1, nN)), delta_x_flat)
        delta_u_final = np.multiply(np.tile(np.eye(nu), (1, nN)), delta_u_flat)
        xx = np.tile(states, (nx, 1)).reshape((nx, nx * nN), order='F')
        # print(delta_x_final, xx)
        ux = np.tile(controls, (nx, 1)).reshape((nu, nx * nN), order='F')
        x_plus = xx + delta_x_final
        # print(x_plus, ux)
        x_minus = xx - delta_x_final
        fx_plus = self.update_dynamics(x_plus, ux, dt)
        # print(fx_plus)
        fx_minus = self.update_dynamics(x_minus, ux, dt)
        A = (fx_plus - fx_minus) / (2 * delta_x_flat)

        xu = np.tile(states, (nu, 1)).reshape((nx, nu * nN), order='F')
        uu = np.tile(controls, (nu, 1)).reshape((nu, nu * nN), order='F')
        u_plus = uu + delta_u_final
        # print(xu)
        u_minus = uu - delta_u_final
        fu_plus = self.update_dynamics(xu, u_plus, dt)
        # print(fu_plus)
        fu_minus = self.update_dynamics(xu, u_minus, dt)
        B = (fu_plus - fu_minus) / (2 * delta_u_flat)

        state_row = np.zeros((nx * nN, nN))
        input_row = np.zeros((nu * nN, nN))
        for ii in range(nN):
            state_row[ii * nx:ii * nx + nx, ii] = states[:, ii]
            input_row[ii * nu:ii * nu + nu, ii] = controls[:, ii]
        d = self.update_dynamics(states, controls, dt) - np.dot(A, state_row) - np.dot(B, input_row)

        return A, B, d

    def form_long_matrices_LTV(self, A, B, d, D):
        nx = 8
        nu = 2
        nl = 8
        N = self.N

        AA = np.zeros((nx * N, nx))
        BB = zeros((nx * N, nu * N))
        dd = zeros((nx * N, 1))
        DD = zeros((nx * N, nl * N))
        AA_i_row = np.eye(nx)
        dd_i_row = np.zeros((nx, 1))
        # B_i_row = zeros((nx, 0))
        # D_i_bar = zeros((nx, nx))
        for ii in np.arange(0, N):
            AA_i_row = np.dot(A[:, :, ii], AA_i_row)
            AA[ii * nx:(ii + 1) * nx, :] = AA_i_row

            B_i_row = B[:, :, ii]
            D_i_row = D[:, :, ii]
            for jj in np.arange(ii - 1, -1, -1):
                B_i_cell = np.dot(A[:, :, ii], BB[(ii - 1) * nx:ii * nx, jj * nu:(jj + 1) * nu])
                B_i_row = np.hstack((B_i_cell, B_i_row))
                D_i_cell = np.dot(A[:, :, ii], DD[(ii - 1) * nx:ii * nx, jj * nl:(jj + 1) * nl])
                D_i_row = np.hstack((D_i_cell, D_i_row))
            BB[ii * nx:(ii + 1) * nx, :(ii + 1) * nu] = B_i_row
            DD[ii * nx:(ii + 1) * nx, :(ii + 1) * nl] = D_i_row

            dd_i_row = np.dot(A[:, :, ii], dd_i_row) + d[:, :, ii]
            dd[ii * nx:(ii + 1) * nx, :] = dd_i_row

        return AA, BB, dd, DD


def train_model():
    time_step = 0.25
    step_rate = int(1000*time_step)

    N0 = int(390/time_step)
    Nf = int(-135/time_step)

    horizon = int(40/time_step)

    mat = scipy.io.loadmat('beta_localhost_2021-11-23-15-28-35_0.mat')
    states = mat['states'][:, ::step_rate]
    controls = mat['inputs'][:, ::step_rate]
    states = states[:, N0:Nf - 1]
    controls = controls[:, N0:Nf - 1]
    time = np.arange(0, states.shape[1]) * time_step
    print(states.shape)

    parametric_model = Model(1)

    num_steps = 1
    f_0 = parametric_model.update_dynamics(states, controls, time_step*num_steps)
    g_hat = states[:, num_steps*1:] - f_0[:, :-1*num_steps]
    print(np.mean(np.abs(g_hat), axis=1))
    D = np.vstack((states[:-3, :-1*num_steps], controls[:, :-1*num_steps]))

    kernel = 1 * gaussian_process.kernels.RBF(length_scale=1.16, length_scale_bounds=(1e-5, 1e5)) + gaussian_process.kernels.WhiteKernel(noise_level=0.248)
    # gp_vx = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
    # gp_vx.fit(D.T, g_hat[0, :])
    # print(gp_vx.kernel_)
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=True)
    t0 = tt.time()
    # gp.optimizer = None
    gp.fit(D[:, :].T, g_hat[:-3, :].T)
    t1 = tt.time()
    print(t1 - t0)
    params = gp.get_params()
    kernel = gp.kernel_
    print(kernel)
    # gp2 = gaussian_process.GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
    # # gp2 = gp2.set_params(params=params)
    # t0 = tt.time()
    # gp2.fit(D[:, :horizon].T, g_hat[:-3, :horizon].T)
    # t1 = tt.time()
    # print(t1 - t0)
    # print(gp2.kernel_)
    # gp = gp2

    predicted_states = np.zeros((8, horizon))
    predicted_state = states[:, :1]
    predicted_states[:, 0:1] = predicted_state
    predicted_states2 = np.zeros((8, horizon))
    predicted_state2 = states[:, :1]
    predicted_states2[:, 0:1] = predicted_state2
    my_kinematics = np.zeros((3, horizon))
    my_kinematic = states[5:, 0:1]
    my_kinematics[:, 0:1] = my_kinematic
    sigmas = np.zeros((5, horizon))
    # print(states[:, ::horizon].shape, controls[:, ::horizon].shape)
    for ii in range(horizon):
        predicted_state = parametric_model.update_dynamics(predicted_state, controls[:, ii:ii+1], time_step)
        predicted_states[:, ii+1:ii+2] = predicted_state
        g_t, g_t_sigma = gp.predict(np.vstack((states[:-3, ii:ii+1], controls[:, ii:ii + 1])).T, return_std=True)
        predicted_state2 = parametric_model.update_dynamics(predicted_state2, controls[:, ii:ii + 1], time_step)
        predicted_state2[:-3, :] += g_t.T
        predicted_states2[:, ii+1:ii + 2] = predicted_state2
        my_kinematic[0, 0] += time_step * predicted_state2[2, 0]
        dot_X = cos(my_kinematic[0, 0]) * predicted_state2[0, 0] - sin(my_kinematic[0, 0]) * predicted_state2[1, 0]
        dot_Y = sin(my_kinematic[0, 0]) * predicted_state2[0, 0] + cos(my_kinematic[0, 0]) * predicted_state2[1, 0]
        my_kinematic[1, 0] += time_step * dot_X
        my_kinematic[2, 0] += time_step * dot_Y
        my_kinematics[:, ii+1:ii+2] = my_kinematic
        A, B, d = parametric_model.linearize_dynamics(predicted_state2, controls[:, ii:ii + 1], time_step)
        sigmas[:, ii+1:ii+2] = g_t_sigma.T**2 + np.diag(np.dot(np.dot(A[:5, :5], np.diag(sigmas[:, ii])), A[:5, :5].T)).reshape((-1, 1))

    g_0_mu, g_0_sigma = gp.predict(D.T, return_std=True)
    g_0_mu = g_0_mu.T
    g_0_sigma = g_0_sigma.T
    print(np.min(g_0_sigma, axis=1), np.mean(g_0_sigma, axis=1), np.max(g_0_sigma, axis=1))

    plt.figure()
    for ii in range(states.shape[0] - 3):
        if ii == 3:
            continue
        elif ii > 3:
            plt.subplot(2, 2, ii)
        else:
            plt.subplot(2, 2, ii + 1)
        plt.plot(time, states[ii, :])
        plt.plot(time[num_steps:], f_0[ii, :-num_steps], 'g')
        # plt.plot(time[:horizon], predicted_states[ii, :], 'g')
        if ii < 5:
            plt.plot(time[num_steps:], f_0[ii, :-num_steps] + g_0_mu[ii, :], 'r')
            plt.fill_between(time[num_steps:], f_0[ii, :-num_steps] + g_0_mu[ii, :] + 2*g_0_sigma[ii, :], f_0[ii, :-num_steps] + g_0_mu[ii, :] - 2*g_0_sigma[ii, :], color='lightcoral')
            # plt.fill_between(time[:horizon], predicted_states2[ii, :] + np.sqrt(sigmas[ii, :]), predicted_states2[ii, :] - np.sqrt(sigmas[ii, :]), color='orchid')
        # else:
            # plt.plot(time[num_steps:], f_0[ii, :-num_steps])
        # plt.plot(time[:horizon], predicted_states2[ii, :], 'm')
        # if ii >= 5:
        #     plt.plot(time[:horizon], my_kinematics[ii-5, :])
    add_labels([0, 1, 2, 4], ('observed', 'nominal', 'GP corrected'))
    # plt.suptitle('Gaussian Processs Training')
    fig = plt.gcf()
    # fig.canvas.manager.window.setGeometry(0, 0, 1920/2, 1080)
    plt.tight_layout()
    fig.canvas.draw()
    plt.pause(0.01)
    plt.savefig('GP Training 95CI_lowres.png', dpi=300)
    # plt.show()

    return gp


def run_model(gp):
    time_step = 0.2
    step_rate = int(1000 * time_step)
    N0 = int((390-240) / time_step)
    Nf = int((-135-240) / time_step)

    horizon = int(40 / time_step)

    mat = scipy.io.loadmat('beta_localhost_2021-11-23-15-28-35_0.mat')
    states = mat['states'][:, ::step_rate]
    controls = mat['inputs'][:, ::step_rate]
    states = states[:, N0:Nf - 1]
    controls = controls[:, N0:Nf - 1]
    time = np.arange(0, states.shape[1]) * time_step
    print(states.shape)

    parametric_model = Model(1)

    num_steps = 1
    f_0 = parametric_model.update_dynamics(states, controls, time_step * num_steps)
    g_hat = states[:, num_steps * 1:] - f_0[:, :-1 * num_steps]
    print(np.mean(np.abs(g_hat), axis=1))
    D = np.vstack((states[:-3, :-1 * num_steps], controls[:, :-1 * num_steps]))

    predicted_states = np.zeros((8, horizon))
    predicted_state = states[:, :1]
    predicted_states[:, 0:1] = predicted_state
    predicted_states2 = np.zeros((8, horizon))
    predicted_state2 = states[:, :1]
    predicted_states2[:, 0:1] = predicted_state2
    my_kinematics = np.zeros((3, horizon))
    my_kinematic = states[5:, 0:1]
    my_kinematics[:, 0:1] = my_kinematic
    sigmas = np.zeros((5, horizon))
    # print(states[:, ::horizon].shape, controls[:, ::horizon].shape)
    for ii in range(horizon):
        predicted_state = parametric_model.update_dynamics(predicted_state, controls[:, ii:ii + 1], time_step)
        predicted_states[:, ii + 1:ii + 2] = predicted_state
        g_t, g_t_sigma = gp.predict(np.vstack((states[:-3, ii:ii + 1], controls[:, ii:ii + 1])).T, return_std=True)
        predicted_state2 = parametric_model.update_dynamics(predicted_state2, controls[:, ii:ii + 1], time_step)
        predicted_state2[:-3, :] += g_t.T
        predicted_states2[:, ii + 1:ii + 2] = predicted_state2
        my_kinematic[0, 0] += time_step * predicted_state2[2, 0]
        dot_X = cos(my_kinematic[0, 0]) * predicted_state2[0, 0] - sin(my_kinematic[0, 0]) * predicted_state2[1, 0]
        dot_Y = sin(my_kinematic[0, 0]) * predicted_state2[0, 0] + cos(my_kinematic[0, 0]) * predicted_state2[1, 0]
        my_kinematic[1, 0] += time_step * dot_X
        my_kinematic[2, 0] += time_step * dot_Y
        my_kinematics[:, ii + 1:ii + 2] = my_kinematic
        A, B, d = parametric_model.linearize_dynamics(predicted_state2, controls[:, ii:ii + 1], time_step)
        sigmas[:, ii + 1:ii + 2] = g_t_sigma.T**2 + np.diag(np.dot(np.dot(A[:5, :5], np.diag(sigmas[:, ii])), A[:5, :5].T)).reshape((-1, 1))

    g_0_mu, g_0_sigma =  gp.predict(D.T, return_std=True)
    g_0_mu = g_0_mu.T
    g_0_sigma = g_0_sigma.T
    print(np.min(g_0_sigma, axis=1), np.mean(g_0_sigma, axis=1), np.max(g_0_sigma, axis=1))

    plt.figure()
    for ii in range(states.shape[0] - 3):
        if ii == 3:
            continue
        elif ii > 3:
            plt.subplot(2, 2, ii)
        else:
            plt.subplot(2, 2, ii + 1)
        plt.plot(time, states[ii, :])
        plt.plot(time[num_steps:], f_0[ii, :-num_steps], 'g')
        # plt.plot(time[:horizon], predicted_states[ii, :], 'g')
        if ii < 5:
            plt.plot(time[num_steps:], f_0[ii, :-num_steps] + g_0_mu[ii, :], 'r')
            plt.fill_between(time[num_steps:], f_0[ii, :-num_steps] + g_0_mu[ii, :] + 2*g_0_sigma[ii, :],
                             f_0[ii, :-num_steps] + g_0_mu[ii, :] - 2*g_0_sigma[ii, :], color='lightcoral')
            # plt.fill_between(time[:horizon], predicted_states2[ii, :] + np.sqrt(sigmas[ii, :]),
            #                  predicted_states2[ii, :] - np.sqrt(sigmas[ii, :]), color='orchid')
        # else:
        #     plt.plot(time[num_steps:], f_0[ii, :-num_steps])
        # plt.plot(time[:horizon], predicted_states2[ii, :], 'm')
        if ii >= 5:
            plt.plot(time[:horizon], my_kinematics[ii - 5, :])
    add_labels([0, 1, 2, 4], ('observed', 'nominal', 'GP corrected'))
    # plt.suptitle('Gaussian Process Validation')
    fig = plt.gcf()
    # fig.canvas.manager.window.setGeometry(1920 / 2, 0, 1920 / 2, 1080)
    plt.tight_layout()
    fig.canvas.draw()
    plt.pause(0.01)
    plt.savefig('GP Validation 95CI_lowres.png', dpi=300)
    plt.show()


class RetraceDrive:

    def __init__(self, file, start, stop, step, horizon):
        self.time_step = step
        step_rate = int(1000 * self.time_step)
        N0 = int((start) / self.time_step)
        Nf = int((stop) / self.time_step)
        self.N = int(horizon / self.time_step)

        self.map = np.load('maps/CCRF/CCRF_2021-01-10.npz')
        mat = scipy.io.loadmat(file)
        states = mat['states'][:, ::step_rate]
        controls = mat['inputs'][:, ::step_rate]
        self.states = states[:, N0:Nf - 1]
        self.controls = controls[:, N0:Nf - 1]
        self.time = np.arange(0, self.states.shape[1]) * self.time_step
        print('loaded states of shape: ', self.states.shape)

        self.parametric_model = Model(self.N)
        kernel = 1 * gaussian_process.kernels.RBF(length_scale=1.16) + gaussian_process.kernels.WhiteKernel(noise_level=0.248)
        self.gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
        self.gp.fit(np.zeros((1, 7)), np.zeros((1, 5)))

    def update_gp(self, observed_states, past_controls, past_nominal_states):
        max_num_obs = 1000
        if observed_states.shape[1] > max_num_obs:
            observed_states = observed_states[:, -max_num_obs-1:]
            past_controls = past_controls[:, -max_num_obs:]
            past_nominal_states = past_nominal_states[:, -max_num_obs-1:]
        D = np.vstack((observed_states[:5, :-1], past_controls)).T
        g_hat = observed_states[:5, 1:].T - past_nominal_states[:5, 1:].T
        self.gp.fit(D, g_hat)

    def rollout_nominal(self, start_state, controls):
        states = np.zeros((start_state.shape[0], self.N+1))
        state = start_state.copy()
        states[:, 0:1] = state
        for ii in range(self.N):
            state = self.parametric_model.update_dynamics(state, controls[:, ii:ii+1], self.time_step)
            states[:, ii + 1:ii + 2] = state
        return states

    def rollout_augmented(self, start_state, controls):
        states = np.zeros((start_state.shape[0], self.N + 1))
        sigmas = np.zeros((start_state.shape[0], self.N + 1))
        state = start_state.copy()
        sigma = np.zeros((8, 8))
        states[:, 0:1] = state
        for ii in range(self.N):
            A, B, d = self.parametric_model.linearize_dynamics(state, controls[:, ii:ii + 1], self.time_step)
            nominal_state = self.parametric_model.update_dynamics(state, controls[:, ii:ii + 1], self.time_step)
            residual_state_mean, residual_state_sigma = self.gp.predict(np.vstack((state[:-3, :], controls[:, ii:ii + 1])).T, return_std=True)
            residual_state_mean = np.vstack((residual_state_mean.T, np.zeros((nominal_state.shape[0] - residual_state_mean.shape[1], 1))))
            state = nominal_state + residual_state_mean
            states[:, ii + 1:ii + 2] = state
            new_cov = np.diag(np.vstack((residual_state_sigma.T, np.zeros((3, 1)))).flatten())
            sigma = new_cov**2 + np.dot(A, np.dot(sigma, A.T))
            sigmas[:, ii + 1:ii + 2] = np.sqrt(np.diag(sigma)).reshape((-1, 1))
        return states, sigmas

    def calc_log_likelihood(self, true_states, predicted_means, predicted_stds):
        likelihoods = scipy.stats.norm.pdf(true_states[:, 2:], predicted_means[:, 2:], predicted_stds[:, 2:])
        log_likelihood = np.sum(np.log(likelihoods + 1e-7))
        return log_likelihood

    def add_labels(self):
        plt.subplot(4, 2, 1)
        plt.gca().legend(('observed vx', 'measured', 'nominal', 'corrected'))
        plt.xlabel('t (s)')
        plt.ylabel('m/s')
        plt.subplot(4, 2, 2)
        plt.gca().legend(('observed vy', 'measured', 'nominal', 'corrected'))
        plt.xlabel('t (s)')
        plt.ylabel('m/s')
        plt.subplot(4, 2, 3)
        plt.gca().legend(('observed wz', 'measured', 'nominal', 'corrected'))
        plt.xlabel('t (s)')
        plt.ylabel('rad/s')
        plt.subplot(4, 2, 4)
        plt.gca().legend(('observerd wF', 'measured', 'nominal', 'corrected'))
        plt.xlabel('t (s)')
        plt.ylabel('rad/s')
        plt.subplot(4, 2, 5)
        plt.gca().legend(('observed wR', 'measured', 'nominal', 'corrected'))
        plt.xlabel('t (s)')
        plt.ylabel('rad/s')
        plt.subplot(4, 2, 6)
        plt.gca().legend(('observed yaw', 'measured', 'nominal', 'corrected'))
        plt.xlabel('t (s)')
        plt.ylabel('rad')
        plt.subplot(4, 2, 7)
        plt.gca().legend(('observed X', 'measured', 'nominal', 'corrected'))
        plt.xlabel('t (s)')
        plt.ylabel('m')
        plt.subplot(4, 2, 8)
        plt.gca().legend(('observed Y', 'measured', 'nominal', 'corrected'))
        plt.xlabel('t (s)')
        plt.ylabel('m')

    def update_plots(self, times, past_states, future_states, nominal_states, corrected_states, sigmas=None):
        # plt.figure()
        plt.clf()
        if self.plot_type == 'states':
            for ii in range(past_states.shape[0]):
                plt.subplot(4, 2, ii + 1)
                plt.plot(times[:past_states.shape[1]], past_states[ii, :], 'b-')
                plt.plot(times[past_states.shape[1] - 1:], future_states[ii, :], 'b.')
                plt.plot(times[past_states.shape[1] - 1:], nominal_states[ii, :], 'g.')
                if sigmas is None:
                    plt.plot(times[past_states.shape[1] - 1:], corrected_states[ii, :], 'r.')
                else:
                    plt.errorbar(times[past_states.shape[1] - 1:], corrected_states[ii, :], yerr=2*sigmas[ii, :], fmt='r.')
                plt.xlim(times[past_states.shape[1]] - 1 * self.N * self.time_step, times[-1])
            self.add_labels()
        elif self.plot_type == 'map':
            plt.plot(past_states[-2,  :], past_states[-1, :], 'b-')
            plt.plot(future_states[-2, :], future_states[-1, :], 'b.')
            plt.plot(nominal_states[-2, :], nominal_states[-1, :], 'g.')
            if sigmas is None:
                plt.plot(corrected_states[-2, :], corrected_states[-1, :], 'r.')
            else:
                plt.errorbar(corrected_states[-2, :], corrected_states[-1, :], yerr=2*sigmas[-1, :], xerr=2*sigmas[-2, :], fmt='r.')
            plt.gca().legend(('observed', 'measured', 'nominal', 'corrected'), prop={'size': 24})
            plt.plot(self.map['X_in'], self.map['Y_in'], 'k')
            plt.plot(self.map['X_out'], self.map['Y_out'], 'k')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
        fig = plt.gcf()
        fig.canvas.draw()
        plt.pause(0.001)
        plt.tight_layout()
        # plt.show()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    def update_map(self, times, past_states, corrected_states, sigmas=None):
        # plt.figure()
        plt.clf()
        plt.plot(past_states[-2,  :], past_states[-1, :], 'b-')
        if sigmas is None:
            pcm = plt.scatter(corrected_states[-2, :], corrected_states[-1, :], c=corrected_states[0, :], marker='.')
            # if self.path.shape[1] < 2:
            self.cbar = plt.colorbar(pcm)
            self.cbar.set_label('speed (m/s)')
            # else:
            #     self.cbar.update_normal(pcm)
        else:
            norm = matplotlib.colors.Normalize(vmin=3.0, vmax=7.0)
            scm = matplotlib.cm.ScalarMappable(norm)
            colors = scm.to_rgba(corrected_states[0, :])
            self.cbar = plt.colorbar(scm)
            self.cbar.set_label('speed (m/s)')
            pcm = plt.scatter(corrected_states[-2, :], corrected_states[-1, :], c=colors, marker='o')
            plt.errorbar(corrected_states[-2, :], corrected_states[-1, :], yerr=2*sigmas[-1, :], xerr=2*sigmas[-2, :], elinewidth=0.3, marker=' ', ecolor=colors)
        plt.gca().legend(('observed', 'planned mean', 'planned uncertainty',), prop={'size': 24})
        plt.plot(self.map['X_in'], self.map['Y_in'], 'k')
        plt.plot(self.map['X_out'], self.map['Y_out'], 'k')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        fig = plt.gcf()
        fig.canvas.draw()
        plt.pause(0.001)
        plt.tight_layout()
        # plt.show()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    def retrace(self):
        past_nominal_states = np.zeros_like(self.states)
        past_nominal_states[:, 0] = self.states[:, 0]
        losses = np.zeros((2, self.states.shape[1]))
        imgs = []
        # plt.figure()
        # fig = plt.gcf()
        # fig.canvas.manager.full_screen_toggle()
        for ii in range(self.states.shape[1] - self.N - 2):
            print(ii/(self.states.shape[1] - self.N - 2))
            t0 = tt.time()
            state = self.states[:, ii:ii + 1]
            controls = self.controls[:, ii:ii + 1 + self.N]
            nominal_states = self.rollout_nominal(state, controls)
            past_nominal_states[:, ii+1] = nominal_states[:, 1]
            corrected_states, uncertainties = self.rollout_augmented(state, controls)
            error = np.linalg.norm(self.states[:3, ii:ii + 1 + self.N] - corrected_states[:3, :])
            log_likelihood = self.calc_log_likelihood(self.states[:, ii:ii + 1 + self.N], corrected_states, uncertainties)
            losses[0, ii] = error
            losses[1, ii] = log_likelihood
            if (ii + 1) % 20 == 0:
                self.gp.optimizer = 'fmin_l_bfgs_b'
            else:
                self.gp.optimizer = None
            self.update_gp(self.states[:, :ii + 2], self.controls[:, :ii + 1], past_nominal_states[:, :ii + 2])
            print(self.gp.kernel_)
            if (ii + 1) % 20 == 0:
                kernel = self.gp.kernel_
                self.gp.kernel = kernel
            print(tt.time() - t0)
        #     image = self.update_plots(self.time[:ii + 1 + self.N], self.states[:, :ii + 1], self.states[:, ii:ii + 1 + self.N], nominal_states, corrected_states, sigmas=uncertainties)
        #     imgs.append(image)
        # imageio.mimsave('drive_retrace.gif', imgs, fps=int(1/self.time_step))
        plt.plot(self.time, losses[0, :])
        plt.plot(self.time, losses[1, :])
        plt.show()

    def laps(self, lap_starts):
        past_nominal_states = np.zeros_like(self.states)
        past_nominal_states[:, 0] = self.states[:, 0]
        losses = np.zeros((2, self.states.shape[1]))
        imgs = []
        plt.figure()
        fig = plt.gcf()
        fig.canvas.manager.full_screen_toggle()
        for ii, start_times in enumerate(lap_starts):
            start_index = int(start_times / self.time_step)
            state = self.states[:, start_index:start_index + 1]
            controls = self.controls[:, start_index:start_index + 1 + self.N]
            nominal_states = self.rollout_nominal(state, controls)
            past_nominal_states = self.parametric_model.update_dynamics(self.states[:, :start_index + 1], self.controls[:, :start_index + 1], dt=self.time_step)
            past_nominal_states = np.hstack((self.states[:, 0:1], past_nominal_states))
            corrected_states, uncertainties = self.rollout_augmented(state, controls)
            # error = np.linalg.norm(self.states[:3, ii:ii + 1 + self.N] - corrected_states[:3, :])
            # log_likelihood = self.calc_log_likelihood(self.states[:, ii:ii + 1 + self.N], corrected_states, uncertainties)
            # losses[0, ii] = error
            # losses[1, ii] = log_likelihood
            self.gp.optimizer = 'fmin_l_bfgs_b'
            self.update_gp(self.states[:, :start_index + 2], self.controls[:, :start_index + 1], past_nominal_states)
            print(self.gp.kernel_)
            # if (ii + 1) % 20 == 0:
            #     kernel = self.gp.kernel_
            #     self.gp.kernel = kernel
            # print(tt.time() - t0)
            image = self.update_plots(self.time[:start_index + 1 + self.N], self.states[:, :start_index + 1], self.states[:, start_index:start_index + 1 + self.N], nominal_states, corrected_states, sigmas=uncertainties)
            # plt.suptitle('lap ' + str(ii+1))
            plt.pause(5.0)
            plt.savefig('lap {} {}_lowres.png'.format(ii+1, self.plot_type), dpi=300)
        #     imgs.append(image)
        # imageio.mimsave('drive_retrace.gif', imgs, fps=int(1/self.time_step))
        # plt.plot(self.time, losses[0, :])
        # plt.plot(self.time, losses[1, :])
        # plt.show()

    def ctrl_design(self, lap_starts):
        solver = cs_solver_obstacle.CSSolver(8, 2, 8, self.N, np.array([[-1.0, 1.0],[-1.0, 1.0]]),
                                             np.array([[-0.1, 0.1],[-0.1,0.1]]), (None,), mean_only=True,
                                             k_form=1, prob_lvl=0.70)
        map = MapCA()
        Q = np.zeros((8, 8))
        Q[0, 0] = 30
        Q[1, 1] = 1
        Q[5, 5] = 10
        Q[6, 6] = 20
        Q_bar = np.kron(np.eye(self.N, dtype=int), Q)
        R = np.zeros((2, 2))
        R[0, 0] = 20  # 2
        R[1, 1] = 2  # 1
        R_bar = np.kron(np.eye(self.N, dtype=int), R)
        x_target = np.tile(np.array([9, 0, 0, 0, 0, 0, 0, 0]).reshape((-1, 1)), (self.N, 1))
        past_nominal_states = np.zeros_like(self.states)
        past_nominal_states[:, 0] = self.states[:, 0]
        losses = np.zeros((2, self.states.shape[1]))
        imgs = []
        plt.figure()
        fig = plt.gcf()
        fig.canvas.manager.full_screen_toggle()
        for ii, start_times in enumerate(lap_starts):
            start_index = int(start_times / self.time_step)
            state = self.states[:, start_index:start_index + 1]
            state[0, 0] = 7.0
            controls = self.controls[:, start_index:start_index + 1 + self.N]
            nominal_states = self.rollout_nominal(state, controls)
            past_nominal_states = self.parametric_model.update_dynamics(self.states[:, :start_index + 1], self.controls[:, :start_index + 1], dt=self.time_step)
            past_nominal_states = np.hstack((self.states[:, 0:1], past_nominal_states))
            corrected_states, uncertainties = self.rollout_augmented(state, controls)
            corrected_states_map = corrected_states.copy()
            for jj in range(self.N):
                corrected_states_map[5:, jj] = map.localize(corrected_states[-2:, jj], corrected_states[-3, jj])
            self.parametric_model.cartesian = False
            A, B, d = self.parametric_model.linearize_dynamics(corrected_states_map[:, :-1], controls[:, :-1], dt=self.time_step)
            self.parametric_model.cartesian = True
            D = np.zeros((8, 8, uncertainties.shape[1]))
            for jj in range(uncertainties.shape[1]):
                D[:, :, jj] = np.diag(uncertainties[:, jj])
            A = A.reshape((8, 8, self.N), order='F')
            B = B.reshape((8, 2, self.N), order='F')
            d = d.reshape((8, 1, self.N), order='F')
            AA, BB, dd, DD = self.parametric_model.form_long_matrices_LTV(A, B, d, D)
            solver.populate_params(AA, BB, dd, DD, corrected_states_map[:, 0:1], np.zeros((8, 8)), np.zeros((8, 8)),
                                  Q_bar, R_bar, controls[:, 0], x_target,
                                  1000*np.ones((8, 1)), track_width=2.0, K=np.zeros((2*self.N, 8*self.N)))
            V, K = solver.solve()
            K = K.reshape((2 * self.N, 8 * self.N))
            X = np.dot(AA, corrected_states_map[:, 0:1]) + np.dot(BB, V.reshape((-1, 1))) + dd
            Sigma_X = np.dot(np.dot((np.eye(8*self.N) + np.dot(BB, K)), (np.dot(DD, DD.T))), (np.eye(8*self.N) + np.dot(BB, K)).T)
            us = V.reshape((2, self.N), order='F')
            xs = X.reshape((8, self.N), order='F')
            xs = np.hstack((corrected_states_map[:, 0:1], xs))
            sigmas_x = np.zeros((8, self.N))
            Sigma_X_diag = np.diag(Sigma_X)
            for jj in range(self.N):
                sigmas_x[:, jj] = Sigma_X_diag[jj*8:8*(jj+1)]
            sigmas_x = np.hstack((np.zeros((8, 1)), np.sqrt(sigmas_x)))
            optimal_trajectory = np.zeros((8, self.N))
            optimal_trajectory, opt_traj_sigma = self.rollout_augmented(corrected_states[:, :1], us)
            self.gp.optimizer = 'fmin_l_bfgs_b'
            self.update_gp(self.states[:, :start_index + 2], self.controls[:, :start_index + 1], past_nominal_states)
            print(self.gp.kernel_)
            # if (ii + 1) % 20 == 0:
            #     kernel = self.gp.kernel_
            #     self.gp.kernel = kernel
            # print(tt.time() - t0)
            image = self.update_map(self.time[:start_index + 1 + self.N], self.states[:, :start_index + 1], optimal_trajectory, opt_traj_sigma)
            # plt.suptitle('lap ' + str(ii+1))
            print('lap: ', ii, 'min speed: ', np.min(optimal_trajectory[0, :]), 'ave speed: ', np.mean(optimal_trajectory[0, :]))
            plt.pause(5.0)
            plt.savefig('lap {} control design_lowres.png'.format(ii+1), dpi=300)
        #     imgs.append(image)
        # imageio.mimsave('drive_retrace.gif', imgs, fps=int(1/self.time_step))
        # plt.plot(self.time, losses[0, :])
        # plt.plot(self.time, losses[1, :])
        # plt.show()


if __name__ == '__main__':
    gp = train_model()
    run_model(gp)
    driver = RetraceDrive('beta_localhost_2021-11-23-15-28-35_0.mat', 111, 272 , 0.05, 5.0)
    driver.plot_type = 'states'
    # driver.retrace()
    driver.laps([142-111, 181-111, 219-111, 255-111])
    driver = RetraceDrive('beta_localhost_2021-11-23-15-28-35_0.mat', 111, 272, 0.05, 2.0)
    driver.plot_type = 'map'
    # driver.retrace()
    driver.laps([130 - 111, 168 - 111, 207 - 111, 243 - 111])
    driver = RetraceDrive('beta_localhost_2021-11-23-15-28-35_0.mat', 111, 272, 0.05, 1.0)
    driver.plot_type = 'map'
    driver.ctrl_design([112 - 111, 151 - 111, 191 - 111, 228 - 111, 268 - 111])