import numpy as np
import matplotlib.pyplot as plt

def state_update(state, control):
    l_r = 0.1
    l_f = 0.1
    alpha = 0
    dt = 0.01
    N = state.shape[1]

    x = state[0, :]
    y = state[1, :]
    psi = state[2, :]
    v = state[3, :]
    beta = state[4, :]

    delta = control[0, :]

    x_dot = v * np.cos(psi + beta)
    y_dot = v * np.sin(psi + beta)
    psi_dot = v/l_r * np.sin(beta)
    v_dot = alpha
    beta = np.arctan(l_r/(l_f + l_r) * np.tan(delta))

    new_state = np.zeros((5, N))
    new_state[0, :] = x + x_dot * dt
    new_state[1, :] = y + y_dot * dt
    new_state[2, :] = psi + psi_dot * dt
    new_state[3, :] = v + v_dot * dt
    new_state[4, :] = beta

    return new_state


# x = Ax + Bu (+ d)
# def linear_state_update(x, u):
#     A = np.zeros((5, 5))
#     B = np.zeros((5, 1))
#
#     A = np.eye(5)
#     A[4, :] = 0
#     #x_new = 1*x
#     A[0, 3] =
#
#     return np.dot(A, x) + np.dot(B, u)


def linearize_dynamics(state, control):
    # Ax + Bu + d = df/dx x + df/du u + x_0
    delta_x = np.diag(np.array([0.0001 ,0.0001, 0.0001, 0.0001, 0.0001]))
    delta_u = np.array([0.0001]).reshape((1, 1))

    x = np.tile(state, (1, 5))
    u = np.tile(control, (1, 5))
    x_plus = x + delta_x
    x_minus = x - delta_x
    f_plus = state_update(x_plus, u)
    f_minus = state_update(x_minus, u)
    A = (f_plus - f_minus) / (2 * 0.0001)

    x = np.tile(state, (1, 1))
    u = np.tile(control, (1, 1))
    u_plus = u + delta_u
    u_minus = u - delta_u
    f_plus = state_update(x, u_plus)
    f_minus = state_update(x, u_minus)
    B = (f_plus - f_minus) / (2 * 0.0001)

    next_state = state_update(state, control)
    d = next_state - np.dot(A, state) - np.dot(B, control)

    return A, B, d


if __name__ == '__main__':
    state = np.array([0, 0, 0, 1, 0]).reshape((-1, 1))
    state2 = state.copy()
    states = np.zeros((5, 1000))
    states2 = np.zeros((5, 1000))
    control = np.array([0.017 * -10]).reshape((-1, 1))
    A, B, d = linearize_dynamics(state, control)
    print(A, B, d)
    for ii in range(1000):
        states[:, ii] = state.flatten()
        states2[:, ii] = state2.flatten()
        state = state_update(state, control)
        A, B, d = linearize_dynamics(state2, control)
        state2 = np.dot(A, state2) + np.dot(B, control) + d
    plt.plot(states[0, :], states[1, :])
    plt.plot(states2[0, :], states2[1, :])
    plt.show()

