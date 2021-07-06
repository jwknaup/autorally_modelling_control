import time
import numpy as np
from numpy import sin, cos, tan, arctan as atan, sqrt, arctan2 as atan2, zeros, zeros_like, abs, pi
import scipy.io
import matplotlib.pyplot as plt
import torch
import throttle_model
from multiprocessing import Process
from multiprocessing.dummy import DummyProcess
import imageio
import hybrid_nn


class MapCA:

    def __init__(self):
        # file_name = 'maps/CCRF/CCRF_2021-01-10.npz'
        file_name = 'maps/CCRF/ccrf_track_optimal.npz'
        track_dict = np.load(file_name)
        try:
            p_x = track_dict['X_cen_smooth']
            p_y = track_dict['Y_cen_smooth']
        except KeyError:
            p_x = track_dict['pts'][:, 0]
            p_y = track_dict['pts'][:, 1]
            self.rho = track_dict['curvature']
        self.p = np.array([p_x, p_y])
        dif_vecs = self.p[:, 1:] - self.p[:, :-1]
        self.dif_vecs = dif_vecs
        self.slopes = dif_vecs[1, :] / dif_vecs[0, :]
        self.midpoints = self.p[:, :-1] + dif_vecs/2
        self.s = np.cumsum(np.linalg.norm(dif_vecs, axis=0))

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
            head_dist -= 2*np.pi
            print(norm_dist, s_dist, head_dist * 180 / np.pi)
        elif head_dist < -np.pi:
            head_dist += 2*np.pi
            print(norm_dist, s_dist, head_dist * 180 / np.pi)
        # if printi:
        #     print(norm_dist, s_dist, head_dist*180/np.pi)
        #     printi=0
        # plt.show()
        return head_dist, norm_dist, s_dist

    def get_cur_reg_from_s(self, s):
        nearest = np.argmin(np.abs(s - self.s))
        x0 = self.p[0, nearest]
        y0 = self.p[1, nearest]
        theta0 = np.arctan2(self.dif_vecs[1, nearest], self.dif_vecs[0, nearest])
        s = self.s[nearest]
        curvature = self.rho[nearest]
        return x0, y0, theta0, s, curvature


class Model:
    def __init__(self, N):
        self.throttle = throttle_model.Net()
        self.throttle.load_state_dict(torch.load('throttle_model1.pth'))
        self.N = N
        self.friction = hybrid_nn.Net()
        self.friction.load_state_dict(torch.load('hybrid_net_ar5.pth'))

        ccrf_curv = np.load('maps/CCRF/ccrf_track_optimal.npz')
        self.s = ccrf_curv['s']
        self.rho = ccrf_curv['curvature']

    def get_curvature(self, s):
        while (s > self.s[-1]).any():
            s[s > self.s[-1]] -= self.s[-1]

        dif = np.abs(s.reshape((-1, 1)) - self.s.reshape((1, -1)))
        idx = np.argmin(dif, axis=1)
        rho = self.rho[idx]
        return rho.flatten()

    def update_dynamics(self, state, input, dt, nn=False, throttle_nn=None, cartesian=np.array([])):
        state = state.copy().T
        input = input.copy().T
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

        # if (vx < 1).any():
        #     vx = np.maximum(vx, 1)
        # if (wF < 10).any():
        #     wF = np.maximum(wF, 10)
        # if (wR < 10).any():
        #     wR = np.maximum(wR, 10)

        m_Vehicle_kSteering = -0.24 # -pi / 180 * 18.7861
        m_Vehicle_kSteering = -0.24 # -pi / 180 * 18.7861
        m_Vehicle_cSteering = -0.02 # 0.0109
        throttle_factor = 0.45
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

            sFx = np.where(wF > 0, (vFx - wF * m_Vehicle_rF) / (wF  * m_Vehicle_rF), 0)
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

            ax = ((fFx * cos(delta) - fFy * sin(delta) + fRx) / m_Vehicle_m + vy * wz)

            next_state = zeros_like(state)
            next_state[:, 0] = vx + deltaT * ((fFx * cos(delta) - fFy * sin(delta) + fRx) / m_Vehicle_m + vy * wz)
            next_state[:, 1] = vy + deltaT * ((fFx * sin(delta) + fFy * cos(delta) + fRy) / m_Vehicle_m - vx * wz)
            vy_dot = ((fFx * sin(delta) + fFy * cos(delta) + fRy) / m_Vehicle_m - vx * wz)
            if nn:
                input_tensor = torch.from_numpy(np.vstack((steering, vx, vy, wz, ax, wF, wR)).T).float()
                # input_tensor = torch.from_numpy(input).float()
                forces = self.friction(input_tensor).detach().numpy()
                fafy = forces[:, 0]
                fary = forces[:, 1]
                fafx= forces[0, 2]
                farx = forces[0, 3]

                next_state[:, 0] = vx + deltaT * ((fafx + farx) / m_Vehicle_m + vy * wz)
                next_state[:, 1] = vy + deltaT * ((fafy + fary) / m_Vehicle_m - vx * wz)
                next_state[:, 2] = wz + deltaT * ((fafy) * m_Vehicle_lF - fary * m_Vehicle_lR) / m_Vehicle_Iz
            else:
                next_state[:, 1] = vy + deltaT * ((fFx * sin(delta) + fFy * cos(delta) + fRy) / m_Vehicle_m - vx * wz)
                next_state[:, 2] = wz + deltaT * (
                            (fFy * cos(delta) + fFx * sin(delta)) * m_Vehicle_lF - fRy * m_Vehicle_lR) / m_Vehicle_Iz
            next_state[:, 3] = wF - deltaT * m_Vehicle_rF / m_Vehicle_IwF * fFx
            if throttle_nn:
                input_tensor = torch.from_numpy(np.hstack((T.reshape((-1, 1)), wR.reshape((-1, 1)) / throttle_factor))).float()
                next_state[:, 4] = wR + deltaT * throttle_nn(input_tensor).detach().numpy().flatten()
            else:
                next_state[:, 4] = T  # wR + deltaT * (m_Vehicle_kTorque * (T-wR) - m_Vehicle_rR * fRx) / m_Vehicle_IwR
            rho = self.get_curvature(Y)
            next_state[:, 5] = psi + deltaT * (wz - (vx * cos(psi) - vy * sin(psi)) / (1 - rho * X) * rho)
            next_state[:, 6] = X + deltaT * (vx * sin(psi) + vy * cos(psi))
            next_state[:, 7] = Y + deltaT * (vx * cos(psi) - vy * sin(psi)) / (1 - rho * X)

            if len(cartesian) > 0:
                cartesian[0, :] += deltaT * wz
                cartesian[1, :] += deltaT * (cos(cartesian[0, :]) * vx - sin(cartesian[0, :]) * vy)
                cartesian[2, :] += deltaT * (sin(cartesian[0, :]) * vx + cos(cartesian[0, :]) * vy)

            t += deltaT
            vx = next_state[:, 0]
            vy = next_state[:, 1]
            wz = next_state[:, 2]
            wF = next_state[:, 3]
            wR = next_state[:, 4]
            psi = next_state[:, 5]
            X = next_state[:, 6]
            Y = next_state[:, 7]

        if len(cartesian) > 0:
            return next_state.T, cartesian
        else:
            return next_state.T

    def linearize_dynamics(self, states, controls, dt=0.1):
        nx = 8
        nu = 2
        nN = self.N

        delta_x = np.array([0.01, 0.001, 0.01, 0.1, 0.1, 0.05, 0.1, 0.2])
        delta_u = np.array([0.001, 0.01])
        delta_x_flat = np.tile(delta_x, (1, nN))
        delta_u_flat = np.tile(delta_u, (1, nN))
        delta_x_final = np.multiply(np.tile(np.eye(nx), (1, nN)), delta_x_flat)
        delta_u_final = np.multiply(np.tile(np.eye(nu), (1, nN)), delta_u_flat)
        xx = np.tile(states, (nx, 1)).reshape((nx, nx*nN), order='F')
        # print(delta_x_final, xx)
        ux = np.tile(controls, (nx, 1)).reshape((nu, nx*nN), order='F')
        x_plus = xx + delta_x_final
        # print(x_plus, ux)
        x_minus = xx - delta_x_final
        fx_plus = self.update_dynamics(x_plus, ux, dt, throttle_nn=self.throttle)
        # print(fx_plus)
        fx_minus = self.update_dynamics(x_minus, ux, dt, throttle_nn=self.throttle)
        A = (fx_plus - fx_minus) / (2 * delta_x_flat)

        xu = np.tile(states, (nu, 1)).reshape((nx, nu*nN), order='F')
        uu = np.tile(controls, (nu, 1)).reshape((nu, nu*nN), order='F')
        u_plus = uu + delta_u_final
        # print(xu)
        u_minus = uu - delta_u_final
        fu_plus = self.update_dynamics(xu, u_plus, dt, throttle_nn=self.throttle)
        # print(fu_plus)
        fu_minus = self.update_dynamics(xu, u_minus, dt, throttle_nn=self.throttle)
        B = (fu_plus - fu_minus) / (2 * delta_u_flat)

        state_row = np.zeros((nx*nN, nN))
        input_row = np.zeros((nu*nN, nN))
        for ii in range(nN):
            state_row[ii*nx:ii*nx + nx, ii] = states[:, ii]
            input_row[ii*nu:ii*nu+nu, ii] = controls[:, ii]
        d = self.update_dynamics(states, controls, dt, throttle_nn=self.throttle) - np.dot(A, state_row) - np.dot(B, input_row)

        return A, B, d

    def form_long_matrices_LTI(self, A, B, D):
        nx = 8
        nu = 2
        N = self.N

        AA = np.zeros((nx*N, nx))
        BB = zeros((nx*N, nu * N))
        DD = zeros((nx, nx * N))
        B_i_row = zeros((nx, 0))
        # D_i_bar = zeros((nx, nx))
        for ii in np.arange(0, N):
            AA[ii*nx:(ii+1)*nx, :] = np.linalg.matrix_power(A, ii+1)

            B_i_cell = np.dot(np.linalg.matrix_power(A, ii), B)
            B_i_row = np.hstack((B_i_cell, B_i_row))
            BB[ii*nx:(ii+1)*nx, :(ii+1)*nu] = B_i_row

            # D_i_bar = np.hstack((np.dot(np.linalg.matrix_power(A, ii - 1), D), D_i_bar))
            # temp = np.hstack((D_i_bar, np.zeros((nx, max(0, nx * N - D_i_bar.shape[1])))))
            # DD = np.vstack((DD, temp[:, 0: nx * N]))

        return AA, BB, DD

    def form_long_matrices_LTV(self, A, B, d, D):
        nx = 8
        nu = 2
        nl = 8
        N = self.N

        AA = np.zeros((nx*N, nx))
        BB = zeros((nx*N, nu * N))
        dd = zeros((nx*N, 1))
        DD = zeros((nx*N, nl * N))
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


def add_labels():
    plt.subplot(7, 2, 1)
    plt.gca().legend(('vx',))
    plt.xlabel('t (s)')
    plt.ylabel('m/s')
    plt.subplot(7, 2, 2)
    plt.gca().legend(('vy',))
    plt.xlabel('t (s)')
    plt.ylabel('m/s')
    plt.subplot(7, 2, 3)
    plt.gca().legend(('wz',))
    plt.xlabel('t (s)')
    plt.ylabel('rad/s')
    plt.subplot(7, 2, 4)
    plt.gca().legend(('wF',))
    plt.xlabel('t (s)')
    plt.ylabel('rad/s')
    plt.subplot(7, 2, 5)
    plt.gca().legend(('wR',))
    plt.xlabel('t (s)')
    plt.ylabel('rad/s')
    plt.subplot(7, 2, 6)
    plt.gca().legend(('e_psi',))
    plt.xlabel('t (s)')
    plt.ylabel('rad')
    plt.subplot(7, 2, 7)
    plt.gca().legend(('e_y',))
    plt.xlabel('t (s)')
    plt.ylabel('m')
    plt.subplot(7, 2, 8)
    plt.gca().legend(('s',))
    plt.xlabel('t (s)')
    plt.ylabel('m')
    plt.subplot(7, 2, 9)
    plt.gca().legend(('Yaw',))
    plt.xlabel('t (s)')
    plt.ylabel('rad')
    plt.subplot(7, 2, 10)
    plt.gca().legend(('X',))
    plt.xlabel('t (s)')
    plt.ylabel('m')
    plt.subplot(7, 2, 11)
    plt.gca().legend(('Y',))
    plt.xlabel('t (s)')
    plt.ylabel('m')
    plt.subplot(7, 2, 12)
    # plt.gca().legend(('s',))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.subplot(7, 2, 13)
    plt.gca().legend(('steering',))
    plt.xlabel('t (s)')
    # plt.ylabel('')
    plt.subplot(7, 2, 14)
    plt.gca().legend(('throttle',))
    plt.xlabel('t (s)')
    # plt.ylabel('m')


def plot(states, controls, sim_length, obs, map):
    # plt.figure()
    plt.clf()
    # time = np.arange(sim_length) / 10
    # mat = scipy.io.loadmat("mppi_data/track_boundaries.mat")
    # inner = mat['track_inner'].T
    # outer = mat['track_outer'].T
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
    # # states = np.load('cs_2Hz_states.npz.npy')
    # # controls = np.load('cs_2Hz_control.npz.npy')
    # # for ii in range(14):
    # #     plt.subplot(7, 2, ii + 1)
    # #     if ii < 11:
    # #         plt.plot(time, states[ii, :])
    # #     elif ii == 11:
    # #         plt.plot(states[-2, :], states[-1, :])
    # #         plt.plot(inner[0, :], inner[1, :], 'k')
    # #         plt.plot(outer[0, :], outer[1, :], 'k')
    # #     else:
    # #         plt.plot(time, controls[ii-12, :])
    # add_labels()
    mat = np.load('maps/CCRF/CCRF_2021-01-10.npz')
    # inner = mat['track_inner'].T
    # outer = mat['track_outer'].T
    # # plt.subplot(7, 2, 12)
    # # plt.plot(inner[0, :], inner[1, :], 'k')
    # # plt.plot(outer[0, :], outer[1, :], 'k')
    # # np.save('cs_5Hz_states', states)
    # # np.save('cs_5Hz_control', controls)
    # plt.show()
    # plt.figure()
    plt.scatter(states[-2, :], states[-1, :], c=states[0, :],  marker='.')
    # states = np.load('cs_5Hz_states.npy')
    # plt.plot(states[-2, :], states[-1, :])
    plt.plot(mat['X_in'], mat['Y_in'], 'k')
    plt.plot(mat['X_out'], mat['Y_out'], 'k')
    # plt.gca().legend(('ltv mpc', 'cs smpc', 'track boundaries'))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    cbar = plt.colorbar()
    cbar.set_label('m/s')

    # map_params = [
    #     [2.78, -2.97, -0.6613, 0, 3.8022, 0],
    #     [10.04, 6.19, 2.4829, 3.8022, 18.3537, 0.1712],
    #     [1.46, 13.11, 2.4829, 22.1559, 11.0228, 0],
    #     [-5.92, 3.80, -0.6613, 33.1787, 18.6666, 0.1683],
    #     [-0.24, -0.66, -0.6613, 51.8453, 7.2218, 0]
    # ]
    # obs = obs[:, :, :, 0]
    # # obs[1, 0, 0] += 0.25
    # reg = map.get_cur_reg_from_s(obs[0, 1, 0])
    # print('reg: ', obs[0,1,0], reg)
    # x0 = reg[0]
    # y0 = reg[1]
    # theta = reg[2]
    # s0 = reg[3]
    # rho = reg[4]
    # pmin = np.zeros((2, obs.shape[0]))
    # pmax = pmin.copy()
    # if rho == 0:
    #     pmin[0, :] = x0 + cos(theta) * (obs[:, 1, 0] - s0)
    #     pmin[1, :] = y0 + sin(theta) * (obs[:, 1, 0] - s0)
    #     pmax[0, :] = x0 + cos(theta) * (obs[:, 1, 1] - s0)
    #     pmax[1, :] = y0 + sin(theta) * (obs[:, 1, 1] - s0)
    #
    #     cart_obs = np.zeros((obs.shape[0], 2, 5))
    #     cart_obs[:, 0, 0] = pmin[0, :] - sin(theta) * obs[:, 0, 0]
    #     cart_obs[:, 1, 0] = pmin[1, :] + cos(theta) * obs[:, 0, 0]
    #     cart_obs[:, 0, 1] = pmin[0, :] - sin(theta) * obs[:, 0, 1]
    #     cart_obs[:, 1, 1] = pmin[1, :] + cos(theta) * obs[:, 0, 1]
    #     cart_obs[:, 0, 2] = pmax[0, :] - sin(theta) * obs[:, 0, 1]
    #     cart_obs[:, 1, 2] = pmax[1, :] + cos(theta) * obs[:, 0, 1]
    #     cart_obs[:, 0, 3] = pmax[0, :] - sin(theta) * obs[:, 0, 0]
    #     cart_obs[:, 1, 3] = pmax[1, :] + cos(theta) * obs[:, 0, 0]
    #     cart_obs[:, 0, 4] = pmin[0, :] - sin(theta) * obs[:, 0, 0]
    #     cart_obs[:, 1, 4] = pmin[1, :] + cos(theta) * obs[:, 0, 0]
    # else:
    #     ds_min = (obs[:, 1, 0] - s0)
    #     ds_max = (obs[:, 1, 1] - s0)
    #     dx_min = 1 / rho * cos(ds_min / (1 / rho)) - 1/rho
    #     dx_max = 1 / rho * cos(ds_max / (1 / rho)) - 1/rho
    #     dy_min = 1 / rho * sin(ds_min / (1 / rho))
    #     dy_max = 1 / rho * sin(ds_max / (1 / rho))
    #     pmin[0, :] = x0 + sin(theta) * dx_min + cos(theta) * dy_min
    #     pmin[1, :] = y0 + sin(theta) * dy_min - cos(theta) * dx_min
    #     pmax[0, :] = x0 + sin(theta) * dx_max + cos(theta) * dy_max
    #     pmax[1, :] = y0 + sin(theta) * dy_max - cos(theta) * dx_max
    #     print('x, y ', sin(theta) * dx_min + cos(theta) * dy_min, sin(theta) * dy_min - cos(theta) * dx_min)
    #
    #     cart_obs = np.zeros((obs.shape[0], 2, 5))
    #     cart_obs[:, 0, 0] = pmin[0, :] - sin(theta + ds_min / (1 / rho)) * obs[:, 0, 0]
    #     cart_obs[:, 1, 0] = pmin[1, :] + cos(theta + ds_min / (1 / rho)) * obs[:, 0, 0]
    #     cart_obs[:, 0, 1] = pmin[0, :] - sin(theta + ds_min / (1 / rho)) * obs[:, 0, 1]
    #     cart_obs[:, 1, 1] = pmin[1, :] + cos(theta + ds_min / (1 / rho)) * obs[:, 0, 1]
    #     cart_obs[:, 0, 2] = pmax[0, :] - sin(theta + ds_max / (1 / rho)) * obs[:, 0, 1]
    #     cart_obs[:, 1, 2] = pmax[1, :] + cos(theta + ds_max / (1 / rho)) * obs[:, 0, 1]
    #     cart_obs[:, 0, 3] = pmax[0, :] - sin(theta + ds_max / (1 / rho)) * obs[:, 0, 0]
    #     cart_obs[:, 1, 3] = pmax[1, :] + cos(theta + ds_max / (1 / rho)) * obs[:, 0, 0]
    #     cart_obs[:, 0, 4] = pmin[0, :] - sin(theta + ds_min / (1 / rho)) * obs[:, 0, 0]
    #     cart_obs[:, 1, 4] = pmin[1, :] + cos(theta + ds_min / (1 / rho)) * obs[:, 0, 0]
    #
    # plt.plot(cart_obs[:, 0, :].T, cart_obs[:, 1, :].T)
    plt.pause(0.001)

    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


def update_opponent_kinematics(obs, dt):
    new_obs = obs.copy()
    new_obs[0, 1, :, :] += 5*dt
    return new_obs


def roll_out_trajectory(x_0, us, N, model, dt, lin=None):
    if lin:
        A, B, d = lin
        xs = np.dot(A, x_0) + np.dot(B, us.flatten().reshape((-1, 1))) + d
        xs = xs.reshape((x_0.shape[0], N), order='F')
        xs = np.hstack((x_0, xs[:, :(N-1)]))
    else:
        xs = np.zeros((x_0.shape[0], N))
        xs[:, 0] = x_0.flatten()
        for ii in range(N - 1):
            xs[:, ii + 1:ii+2], cartesian = model.update_dynamics(xs[:, ii:ii+1], us[:, ii:ii+1], dt, throttle_nn=model.throttle, cartesian=xs[5:, ii:ii+1])
            xs[5:, ii + 1:ii+2] = cartesian
        # print(x_0, us, xs)
    return xs


def run_simple_controller():
    plt.ion()
    n = 8
    m = 2
    l = 8
    N = 20
    dt = 0.05
    ar = Model(N)
    map = MapCA()
    x_target = np.tile(np.array([10, 0, 0, 0, 0, 0, 0, 0]).reshape((-1, 1)), (N-1, 1))
    x_target = np.vstack((x_target, np.array([2, 0, 0, 0, 0, 0, 0, 0]).reshape((-1, 1))))
    state = np.asarray([[2, 0.0, 0.0, 20, 20, 0., 5.5, -2.86]]).T
    # state = np.asarray([[5, 0.0, 0.0, 20, 20, 2.35, 9, -35]]).T
    # state = np.asarray([[2, 0.0, 0.0, 20, 20, 2.0, -7, -21]]).T
    e_psi, e_n, s = map.localize(np.asarray([state[6, 0], state[7, 0]]), state[5, 0])
    x = state.copy()
    x[5:, :] = np.vstack((e_psi, e_n, s))
    cartesian = state[5:, :]
    u = np.array([0.0001, 0.3]).reshape((2, 1))
    xs = np.tile(x, (1, N))
    us = np.tile(u, (1, N))
    v_range = np.array([[-1, 1], [.1, 1]])
    slew_rate = np.array([[-0.4, 0.4], [-0.1, 0.1]])
    sigma_0 = np.zeros((n, n))
    # sigma_0 = np.random.randn(n, n)
    # sigma_0 = np.dot(sigma_0, sigma_0.T)
    # sigma_0 = np.eye(n)
    sigma_N_inv = sigma_0
    Q = np.zeros((n, n))
    Q[0, 0] = 3
    Q[1, 1] = 3
    Q[2, 2] = 0.1
    Q[5, 5] = 100
    Q[6, 6] = 200
    Q[7, 7] = 0
    Q_bar = np.kron(np.eye(N, dtype=int), Q)
    # Q_bar[-n, -n] = 3
    # Q_bar[-n+5, -n+5] = 10
    # Q_bar[-n + 6, -n + 6] = 10
    Q_bar[-n, -n] = 30
    Q_bar[-7, -7] = 1000
    # Q_bar[-6, -6] = 100
    Q_bar[-3, -3] = 1000
    Q_bar[-2, -2] = 2000
    Q_bar[-1, -1] = 0
    R = np.zeros((m, m))
    R[0, 0] = 10
    R[1, 1] = 5
    R_bar = np.kron(np.eye(N, dtype=int), R)
    D = np.zeros((n, l))
    mu_N = 10000*np.array([5., 2., 4., 60., 60., 4., 4., 300.]).reshape((-1, 1))

    sim_length = int(26/dt)
    states = np.zeros((8+3, sim_length))
    controls = np.zeros((2, sim_length))

    # ks = np.zeros((m*N, n*N, sim_length))
    # ss = np.zeros((n, sim_length))
    dictionary = np.load("Ks_ltv_20N_5mps_ccrf.npz")
    ks = dictionary['ks']
    ss = dictionary['ss']

    obs = np.array([[[0.1, 0.3], [4.5, 5.5]], ])
    obs = obs.reshape((1, 2, 2, 1))
    obs = np.tile(obs, (1, 1, 1, N))
    for ii in range(N):
        obs[0, 1, :, ii] += 5*0.1*ii
    solver = CSSolver(n, m, l, N, v_range, slew_rate, (False, 4*N), mean_only=False, k_form=1, prob_lvl=0.80, chance_const_N=10)
    # solver.M.setSolverParam("mioTolAbsGap", 100)
    # solve_process = DummyProcess(target=solver.solve)
    # obs = np.swapaxes(obs, 2, 3)
    # obs = np.swapaxes(obs, 1, 3)
    # obs = obs.flatten()
    imgs = []
    try:
        for ii in range(int(sim_length/1)):
            if ii == 76:
                print()
            t0 = time.time()
            A, B, d = ar.linearize_dynamics(xs, us, dt=dt)
            if B[4, 1] < 0:
                print(xs, us)
            # print(d)
            A = A.reshape((n, n, N), order='F')
            B = B.reshape((n, m, N), order='F')
            d = d.reshape((n, 1, N), order='F')
            # D = np.eye(n)
            # D[1, 1] = 0.001
            # D[2, 2] = 0.001
            D = np.tile(D.reshape((n, l, 1)), (1, 1, N))
            # A = np.eye(8)
            # A[0, 0] = 0.5
            # A[0, 4] = 0.5
            # A[4, 4] = 0.9
            # A[5, 2] = 1
            # B = np.zeros((8, 2))
            # B[2, 0] = 1
            # B[4, 1] = 1
            # B = np.vstack((np.hstack((B, np.zeros_like(B), np.zeros_like(B))), np.hstack((np.dot(A, B), B, np.zeros_like(B))), np.hstack((np.dot(A, np.dot(A, B)), np.dot(A, B), B))))
            # A = np.vstack((A, np.dot(A, A), np.dot(A, np.dot(A, A))))
            A, B, d, D = ar.form_long_matrices_LTV(A, B, d, D)
            # A, B, D = ar.form_long_matrices_LTI(A[:, :, 0], B[:, :, 0], np.zeros((8, 8)))
            # print(np.allclose(A1, A), np.allclose(B1, B))
            nearest = np.argmin(np.linalg.norm(state[:, 0:1] - ss[:, :250], axis=0))
            K = ks[:, :, nearest]
            K = np.zeros((m*N, n*N))
            s_targets = xs[7, 0] + np.cumsum(x_target[0::n, 0] * dt * 2)
            s_targets = np.where(np.abs(s_targets - xs[7, :]) > 100, xs[7, :] + 0.5, s_targets)
            # s_targets[np.abs(s_targets - xs[7, :]) > 100] = 183
            x_target[7::n, 0] = s_targets
            solver.populate_params(A, B, d, D, xs[:, 0], sigma_0, sigma_N_inv, Q_bar, R_bar, us[:, 0], x_target,
                                   mu_N, 2.0, K=K, obs=obs)  # np.zeros((m*N, n*N))
            # A = np.eye(8)
            # A[0, 0] = 0.5
            # A[0, 4] = 0.5
            # A[5, 2] = 1
            # B = np.zeros((8, 2))
            # B[2, 0] = 1
            # B[4, 1] = 0.01
            # solver.M.setSolverParam("numThreads", 8)
            # solver.M.setSolverParam("intpntCoTolPfeas", 1e-3)
            # solver.M.setSolverParam("intpntCoTolDfeas", 1e-3)
            # solver.M.setSolverParam("intpntCoTolRelGap", 1e-3)
            # solver.M.setSolverParam("intpntCoTolInfeas", 1e-3)
            # X = np.dot(A, x)
            # print(X.reshape((10, 8)))
            # solve_process.start()
            # solve_process.join()
            # V, K = (solver.V.level(), solver.K.level())
            # try:
            V, K = solver.solve()
            K = K.reshape((m*N, n*N))
            # except RuntimeError:
            #     V = np.tile(np.array([0, 0.1]).reshape((-1, 1)), (N, 1)).flatten()
            #     K = np.zeros((m*N, n*N))
            # ks[:, :, ii] = K[:, :]
            # ss[:, ii] = state[:, 0]
            # nearest = np.argmin(np.abs(state[7, 0] - ss[0, :]))
            # K = ks[:, :, nearest]
            us = V.reshape((m, N), order='F')
            us[:, 0] = V[:m]
            t = 0
            print(xs[:, 0])
            print(us[:, 0])
            print(ar.get_curvature(xs[7,0]))
            X_bar = np.dot(A, xs[:, 0]) + np.dot(B, V) + d.flatten()
            y = np.zeros((n, 1)).flatten()
            for jj in range(1):
                # print(y)
                u = V[jj*m:(jj+1)*m] #+ np.dot(K[jj*m:(jj+1)*m, jj*n:(jj+1)*n], y)
                u = np.where(u > v_range[:, 1], v_range[:, 1], u)
                u = np.where(u < v_range[:, 0], v_range[:, 0], u)
                states[:n, ii*1+jj] = state.flatten()
                states[n:, ii*1+jj] = cartesian.flatten()
                controls[:, ii*1+jj] = u
                # print(state)
                # print(u)
                state, cartesian = ar.update_dynamics(state, u.reshape((-1, 1)), dt, throttle_nn=ar.throttle, cartesian=cartesian)
                state[5:] = cartesian
                # state += np.array([0.1, 0.01, 0.01, 1, 1, 0, 0, 0]).reshape((-1, 1)) * np.random.randn(n, 1)
                # y = state.flatten() - X_bar[jj * n:(jj + 1) * n]
                if jj == 0:
                    D = np.diag(y)
            # us[:, 0:1] += Ky
            # us[:, 0] = np.where(us[:, 0] > u_max, u_max, us[:, 0])
            # us[:, 0] = np.where(us[:, 0] < u_min, u_min, us[:, 0])
            print(us[:, 0])
            obs = update_opponent_kinematics(obs, dt)
            # state = ar.update_dynamics(state, us[:, 0:1], 0.05, throttle_nn=ar.throttle)
            xs = np.dot(A, xs[:, 0]) + np.dot(B, V) + d.flatten()
            xs = xs.reshape((n, N), order='F')
            # y = xs[:, 0:1].copy()
            xs[:, 0] = state.flatten()
            # e_psi, e_n, s = map.localize(np.asarray([xs[6, 0], xs[7, 0]]), xs[5, 0])
            # xs[5:, 0:1] = np.vstack((e_psi, e_n, s))
            xs = roll_out_trajectory(xs[:, 0], us, N, ar, dt)
            for jj in range(N):
                e_psi, e_n, s = map.localize(np.asarray([xs[6, jj], xs[7, jj]]), xs[5, jj])
                xs[5:, jj:jj+1] = np.vstack((e_psi, e_n, s))
            # print('xs', xs)
            y = xs[:, 0] - X_bar[0 * n:(0 + 1) * n]
            D = np.diag(y)
            # print(xs)
            # X = np.dot(A, x) + np.dot(B, V.reshape((-1, 1)))
            # print(X.reshape((10, 8)))
            #     solver.time()
            print(time.time() - t0)

            img = plot(states[:, :ii], controls[:, :ii], ii, obs, map)
            imgs.append(img)

        imageio.mimsave('ccrf_animation_speed.gif', imgs, fps=int(1/dt))
        # plt.plot(pmin[0, 0], pmin[1, 0], 'o')
        # plt.plot(pmax[0, 0], pmax[1, 0], 'o')
        # plt.plot(cart_obs[0, 0, 0], cart_obs[0, 1, 0], 'x')
        # print(obs[0, :, 0])

        # plt.legend(("trajectory", "obstacle 1", "obstacle 2"))
        # plt.gca().add_patch(rect)
        # plt.show()
        # np.savez('Ks_ltv_20N_5mps_ccrf.npz', ks=ks, ss=ss)
    finally:
        solver.M.dispose()


from cs_solver_obstacle import CSSolver
if __name__ == '__main__':
    # u_min = np.array([-0.9, 0.1])
    # u_max = np.array([0.9, 0.9])
    # x = np.array([4., 0., 0., 50., 50., 0.1, 0., 0.]).reshape((8, 1))
    # u = np.array([0.01, 0.5]).reshape((2, 1))
    # xs = np.tile(x, (1, 10))
    # us = np.tile(u, (1, 10))
    # solver = CSSolver(8, 2, 8, 10, u_min, u_max)
    # solver2 = CSSolver(8, 2, 8, 10, u_min, u_max)
    # ar = Model(10)
    # solve_process = DummyProcess(target=solver.solve)
    # solve_process2 = DummyProcess(target=solver2.solve)
    # lin_process = DummyProcess(target=ar.linearize_dynamics, args=(xs, us))
    # solve_process.start()
    # # lin_process.start()
    # solve_process.join()
    # # lin_process.join()
    # t0 = time.time()
    # A, B, d = ar.linearize_dynamics(xs, us)
    # D = np.zeros((8, 8, 10))
    # A = A.reshape((8, 8, 10), order='F')
    # B = B.reshape((8, 2, 10), order='F')
    # d = d.reshape((8, 1, 10), order='F')
    # ar.form_long_matrices_LTV(A, B, d, D)
    # print(time.time() - t0)
    # ar = Model(1)
    # throttle = throttle_model.Net()
    # xs = np.array([4.97, -0.01, -0.069, 50, 49, 0.1, 0.0049, 0.049]).reshape((-1, 1))
    # us = np.array([-0.06377, 0.1]).reshape((-1, 1))
    # A, B, d = ar.linearize_dynamics(xs, us)
    # print(A, B)
    # # x1 = ar.update_dynamics(xs, us, 0.1, throttle_nn=ar.throttle)
    # # print(x1)
    # T = np.array([.05, 0.1, 0.15])
    # wR = np.array([49, 49, 49])
    # throttle_factor = 0.31
    # input_tensor = torch.from_numpy(np.hstack((T.reshape((-1, 1)), wR.reshape((-1, 1)) / throttle_factor))).float()
    # dwR = ar.throttle(input_tensor).detach().numpy().flatten()
    # print(dwR)
    run_simple_controller()
    # n = 8
    # m=2
    # N=10
    # x = np.array([5., 0., 0., 50., 50., 0., 0., 0.]).reshape((8, 1))
    # u = np.array([0.02, 50.]).reshape((2, 1))
    # xs = np.tile(x, (1, 1))
    # us = np.tile(u, (1, 1))
    # A, B, d = linearize_dynamics(xs, us)
    # sigma_0 = np.dot(np.dot(A, x), np.dot(A, x).T) - np.dot(x, x.T)
    # # print(sigma_0)
    # # print(A, B)
    # A, B, D = form_long_matrices_LTI(A, B, np.zeros((8,8)))
    #
    # sigma_0 = np.random.randn(n, n)
    # sigma_0 = np.dot(sigma_0, sigma_0.T)
    # sigma_0 = np.zeros((n, n))
    # # sigma_N_inv = np.linalg.inv(1e6 * sigma_0)
    # sigma_N_inv = sigma_0
    # Q = np.zeros((n, n))
    # Q[0,0] = 1000
    # Q[5,5] = 10
    # Q[6, 6] = 10
    # Q_bar = np.kron(np.eye(N,dtype=int),Q)
    # R = np.zeros((m, m))
    # R[0,0] = 10
    # R[1,1] = 0.00001
    # R_bar = np.kron(np.eye(N,dtype=int),R)
    #
    # solver = CSSolver()
    # try:
    #     solver.populate_params(A, B, D, x, sigma_0, sigma_N_inv, Q_bar, R_bar)
    #     # solver.M.setSolverParam("numThreads", 8)
    #     # solver.M.setSolverParam("intpntCoTolPfeas", 1e-3)
    #     # solver.M.setSolverParam("intpntCoTolDfeas", 1e-3)
    #     # solver.M.setSolverParam("intpntCoTolRelGap", 1e-3)
    #     # solver.M.setSolverParam("intpntCoTolInfeas", 1e-3)
    #     # X = np.dot(A, x)
    #     # print(X.reshape((10, 8)))
    #     V = solver.solve()
    #     X = np.dot(A, x) + np.dot(B, V.reshape((-1, 1)))
    #     print(X.reshape((10, 8)))
    #     #     solver.time()
    # finally:
    #     solver.M.dispose()
