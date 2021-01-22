import numpy as np
import scipy.io
import itertools
import matplotlib.pyplot as plt
import pandas
from numpy import sin, cos, tan, arctan as atan, sqrt, arctan2 as atan2, zeros, zeros_like, abs, pi


def update_dynamics(state, input, nn=None):
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

    m_Vehicle_kSteering = 18.7861
    m_Vehicle_cSteering = 0.0109
    # delta = input[:, 0]
    steering = input[0, 0]
    delta = m_Vehicle_kSteering * steering + m_Vehicle_cSteering
    # T = m_Vehicle_kThrottle * input[:, 1]

    min_velo = 0.1
    deltaT = 0.01

    beta = atan2(vy, vx)

    V = sqrt(vx * vx + vy * vy)
    vFx = V * cos(beta - delta) + wz * m_Vehicle_lF * sin(delta)
    vFy = V * sin(beta - delta) + wz * m_Vehicle_lF * cos(delta)
    vRx = vx
    vRy = vy - wz * m_Vehicle_lR

    sEF = -(vFx - wF * m_Vehicle_rF) / (vFx) + tire_Sh
    muFx = tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv
    sEF = -(vRx - wR * m_Vehicle_rR) / (vRx) + tire_Sh
    muRx = tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv

    sEF = atan(vFy / abs(vFx)) + tire_Sh
    alpha = -sEF
    muFy = -tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv
    sEF = atan(vRy / abs(vRx)) + tire_Sh
    alphaR = -sEF
    muRy = -tire_D * sin(tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))) + tire_Sv

    fFz = m_Vehicle_m * m_g * (m_Vehicle_lR - m_Vehicle_h * muRx) / (
            m_Vehicle_lF + m_Vehicle_lR + m_Vehicle_h * (muFx * cos(delta) - muFy * sin(delta) - muRx))
    # fFz = m_Vehicle_m * m_g * (m_Vehicle_lR / 0.57)
    fRz = m_Vehicle_m * m_g - fFz

    fFx = fFz * muFx
    fRx = fRz * muRx
    fFy = fFz * muFy
    fRy = fRz * muRy

    ax = ((fFx * cos(delta) - fFy * sin(delta) + fRx) / m_Vehicle_m + vy * wz)

    dot_X =cos(psi)*vx - sin(psi)*vy
    dot_Y = sin(psi)*vx + cos(psi)*vy

    next_state = zeros_like(state)
    next_state[:, 0] = vx + deltaT * ((fFx * cos(delta) - fFy * sin(delta) + fRx) / m_Vehicle_m + vy * wz)
    next_state[:, 1] = vy + deltaT * ((fFx * sin(delta) + fFy * cos(delta) + fRy) / m_Vehicle_m - vx * wz)
    vy_dot = ((fFx * sin(delta) + fFy * cos(delta) + fRy) / m_Vehicle_m - vx * wz)
    if nn:
        input_tensor = torch.from_numpy(np.vstack((steering, vx, vy, wz, ax, wF, wR)).T).float()
        # input_tensor = torch.from_numpy(input).float()
        forces = nn(input_tensor).detach().numpy()
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
    # next_state[:, 4] = wR + deltaT * (m_Vehicle_kTorque * (T-wR) - m_Vehicle_rR * fRx) / m_Vehicle_IwR
    next_state[:, 5] = psi + deltaT * wz
    next_state[:, 6] = X + deltaT * dot_X
    next_state[:, 7] = Y + deltaT * dot_Y

    # print(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4)

    return next_state


def run_model():
    mat = scipy.io.loadmat('mppi_data/mppi_states_controls1.mat')
    states = mat['states'][:, ::10].T
    controls = mat['inputs'][:, ::10].T

    # mat = scipy.io.loadmat('mppi_data/mppi_ff_training1.mat')
    # # measured_states1 = mat['ff'][6:, N0:Nf]
    # # dx, N1 = measured_states1.shape
    # controls1 = mat['ff'][:, N0:Nf].T
    # print(controls1.shape)
    # # du, N1 = controls1.shape
    # # forces = mat['ff'][6:, N0:Nf].T

    states = states[N0:Nf-1, :]
    controls = controls[N0:Nf-1, :]
    print(controls.shape)
    time = np.arange(0, len(states)) * 0.01
    analytic_states = np.zeros_like(states)
    nn_states = np.zeros_like(states)
    state1 = states[0:1, :]
    state2 = states[0:1, :]
    for ii in range(len(time)):
        analytic_states[ii, :] = state1
        nn_states[ii, :] = state2
        state1 = update_dynamics(state1, controls[ii:ii+1, :])
        state2 = update_dynamics(state2, controls[ii:ii+1, :], dyn_model)
        state1[:, 4:5] = states[ii, 4:5]
        state2[:, 4:5] = states[ii, 4:5]
        # state2[:, 0] = states[ii, 0]
        # state1[:, 0:2] = states[ii, 0:2]
        # state2[:, 0:2] = states[ii, 0:2]
    plt.figure()
    for ii in range(states.shape[1]):
        plt.subplot(4,2,ii+1)
        plt.plot(time, states[:, ii])
        plt.plot(time, analytic_states[:, ii])
        plt.plot(time, nn_states[:, ii])
    add_labels()
    plt.show()


if __name__ == '__main__':

