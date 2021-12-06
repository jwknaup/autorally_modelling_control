import numpy as np
from numpy import sin, cos, tan, arctan as atan, sqrt, arctan2 as atan2, zeros, zeros_like, abs, pi
import scipy.io
import matplotlib.pyplot as plt
import torch
from sklearn import gaussian_process

import throttle_model


class Model:
    def __init__(self, N):
        self.throttle = throttle_model.Net()
        self.throttle.load_state_dict(torch.load('throttle_model1.pth'))
        self.N = N

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
            next_state[:, 5] = psi + deltaT * wz
            next_state[:, 6] = X + deltaT * dot_X
            next_state[:, 7] = Y + deltaT * dot_Y

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


def train_model():
    time_step = 0.05
    step_rate = int(1000*time_step)

    N0 = int(390/time_step)
    Nf = int(-135/time_step)

    horizon = int(20/time_step)

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
    gp.fit(D.T, g_hat[:-3, :].T)
    print(gp.kernel_)

    predicted_states = np.zeros((8, horizon))
    predicted_state = states[:, :1]
    predicted_states[:, 0:1] = predicted_state
    predicted_states2 = np.zeros((8, horizon))
    predicted_state2 = states[:, :1]
    predicted_states2[:, 0:1] = predicted_state2
    my_kinematics = np.zeros((3, horizon))
    my_kinematic = states[5:, 0:1]
    my_kinematics[:, 0:1] = my_kinematic
    # print(states[:, ::horizon].shape, controls[:, ::horizon].shape)
    for ii in range(horizon):
        predicted_state = parametric_model.update_dynamics(predicted_state, controls[:, ii:ii+1], time_step)
        predicted_states[:, ii+1:ii+2] = predicted_state
        g_t = gp.predict(np.vstack((states[:-3, ii:ii+1], controls[:, ii:ii + 1])).T)
        predicted_state2 = parametric_model.update_dynamics(predicted_state2, controls[:, ii:ii + 1], time_step)
        predicted_state2[:-3, :] += g_t.T
        predicted_states2[:, ii+1:ii + 2] = predicted_state2
        my_kinematic[0, 0] += time_step * predicted_state2[2, 0]
        dot_X = cos(my_kinematic[0, 0]) * predicted_state2[0, 0] - sin(my_kinematic[0, 0]) * predicted_state2[1, 0]
        dot_Y = sin(my_kinematic[0, 0]) * predicted_state2[0, 0] + cos(my_kinematic[0, 0]) * predicted_state2[1, 0]
        my_kinematic[1, 0] += time_step * dot_X
        my_kinematic[2, 0] += time_step * dot_Y
        my_kinematics[:, ii+1:ii+2] = my_kinematic

    g_0_mu, g_0_sigma =  gp.predict(D.T, return_cov=True)
    g_0_mu = g_0_mu.T

    plt.figure()
    for ii in range(states.shape[0]):
        plt.subplot(4, 2, ii + 1)
        plt.plot(time, states[ii, :])
        plt.plot(time[num_steps:], f_0[ii, :-num_steps])
        plt.plot(time[:horizon], predicted_states[ii, :])
        if ii < 5:
            plt.plot(time[num_steps:], f_0[ii, :-num_steps] + g_0_mu[ii, :])
            plt.errorbar(time[num_steps:], f_0[ii, :-num_steps] + g_0_mu[ii, :], yerr=g_0_cov[ii, :])
        else:
            plt.plot(time[num_steps:], f_0[ii, :-num_steps])
        plt.plot(time[:horizon], predicted_states2[ii, :])
        if ii >= 5:
            plt.plot(time[:horizon], my_kinematics[ii-5, :])
    # add_labels()
    plt.show()

    return gp


def run_model(gp):
    time_step = 0.05
    step_rate = int(1000 * time_step)
    N0 = int((390-240) / time_step)
    Nf = int((-135-120) / time_step)

    horizon = int(60 / time_step)

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
    # print(states[:, ::horizon].shape, controls[:, ::horizon].shape)
    for ii in range(horizon):
        predicted_state = parametric_model.update_dynamics(predicted_state, controls[:, ii:ii + 1], time_step)
        predicted_states[:, ii + 1:ii + 2] = predicted_state
        g_t = gp.predict(np.vstack((states[:-3, ii:ii + 1], controls[:, ii:ii + 1])).T)
        predicted_state2 = parametric_model.update_dynamics(predicted_state2, controls[:, ii:ii + 1], time_step)
        predicted_state2[:-3, :] += g_t.T
        predicted_states2[:, ii + 1:ii + 2] = predicted_state2
        my_kinematic[0, 0] += time_step * predicted_state2[2, 0]
        dot_X = cos(my_kinematic[0, 0]) * predicted_state2[0, 0] - sin(my_kinematic[0, 0]) * predicted_state2[1, 0]
        dot_Y = sin(my_kinematic[0, 0]) * predicted_state2[0, 0] + cos(my_kinematic[0, 0]) * predicted_state2[1, 0]
        my_kinematic[1, 0] += time_step * dot_X
        my_kinematic[2, 0] += time_step * dot_Y
        my_kinematics[:, ii + 1:ii + 2] = my_kinematic

    plt.figure()
    for ii in range(states.shape[0]):
        plt.subplot(4, 2, ii + 1)
        plt.plot(time, states[ii, :])
        plt.plot(time[num_steps:], f_0[ii, :-num_steps])
        plt.plot(time[:horizon], predicted_states[ii, :])
        if ii < 5:
            plt.plot(time[num_steps:], f_0[ii, :-num_steps] + gp.predict(D.T).T[ii, :])
        else:
            plt.plot(time[num_steps:], f_0[ii, :-num_steps])
        plt.plot(time[:horizon], predicted_states2[ii, :])
        if ii >= 5:
            plt.plot(time[:horizon], my_kinematics[ii - 5, :])
    # add_labels()
    plt.show()


if __name__ == '__main__':
    gp = train_model()
    run_model(gp)
