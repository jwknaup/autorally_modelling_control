from multiprocessing.dummy import DummyProcess, Queue
import numpy as np
import matplotlib.pyplot as plt
import cs_model
import cs_solver
import rospy
from autorally_private_msgs.msg import mapCA
from autorally_msgs.msg import chassisCommand


def add_labels():
    plt.subplot(5, 2, 1)
    plt.gca().legend(('vx',))
    plt.xlabel('t (s)')
    plt.ylabel('m/s')
    plt.subplot(5, 2, 2)
    plt.gca().legend(('vy',))
    plt.xlabel('t (s)')
    plt.ylabel('m/s')
    plt.subplot(5, 2, 3)
    plt.gca().legend(('wz',))
    plt.xlabel('t (s)')
    plt.ylabel('rad/s')
    plt.subplot(5, 2, 4)
    plt.gca().legend(('wF',))
    plt.xlabel('t (s)')
    plt.ylabel('rad/s')
    plt.subplot(5, 2, 5)
    plt.gca().legend(('wR',))
    plt.xlabel('t (s)')
    plt.ylabel('rad/s')
    plt.subplot(5, 2, 6)
    plt.gca().legend(('e_psi',))
    plt.xlabel('t (s)')
    plt.ylabel('rad')
    plt.subplot(5, 2, 7)
    plt.gca().legend(('e_y',))
    plt.xlabel('t (s)')
    plt.ylabel('m')
    plt.subplot(5, 2, 8)
    plt.gca().legend(('s',))
    plt.xlabel('t (s)')
    plt.ylabel('m')
    plt.subplot(5, 2, 9)
    plt.gca().legend(('steering',))
    plt.xlabel('t (s)')
    # plt.ylabel('')
    plt.subplot(5, 2, 10)
    plt.gca().legend(('throttle',))
    plt.xlabel('t (s)')
    # plt.ylabel('m')


def plot(states, controls, sim_length):
    plt.figure()
    time = np.arange(sim_length) / 10
    for ii in range(10):
        if ii < 8:
            plt.subplot(5, 2, ii + 1)
            plt.plot(time, states[ii, :])
        else:
            plt.subplot(5, 2, ii + 1)
            plt.plot(time, controls[ii-8, :])
    add_labels()
    # mat = scipy.io.loadmat('mppi_data/track_boundaries.mat')
    # inner = mat['track_inner'].T
    # outer = mat['track_outer'].T
    # plt.subplot(7, 2, 12)
    # plt.plot(inner[0, :], inner[1, :], 'k')
    # plt.plot(outer[0, :], outer[1, :], 'k')
    plt.show()


class CS_SMPC:
    def __init__(self):
        rospy.init_node('cs_smpc')
        rospy.Subscriber("/MAP_CA/mapCA", mapCA, self.map_callback)
        self.command_pub = rospy.Publisher("/LTVMPC/chassisCommand", chassisCommand, queue_size=1)
        self.chassis_command = chassisCommand()
        self.begin = 0

        self.n = 8
        self.m = 2
        self.l = 8
        self.N = 10
        self.dt_linearization = 0.1
        self.dt_solve = 0.5

        self.ar = cs_model.Model(self.N)
        self.u_min = np.array([-0.9, 0.1])
        self.u_max = np.array([0.9, 0.9])
        self.solver = cs_solver.CSSolver(self.n, self.m, self.l, self.N, self.u_min, self.u_max)

        Q = np.zeros((self.n, self.n))
        Q[0, 0] = 3
        Q[1, 1] = 1
        Q[5, 5] = 10
        Q[6, 6] = 10
        self.Q_bar = np.kron(np.eye(self.N, dtype=int), Q)
        R = np.zeros((self.m, self.m))
        R[0, 0] = 10
        R[1, 1] = 0.001
        self.R_bar = np.kron(np.eye(self.N, dtype=int), R)

        self.state = np.zeros((self.n, 1))
        self.control = np.zeros((self.m, 1))

    def map_callback(self, map_msg):
        vx = map_msg.vx
        vy = map_msg.vy
        wz = map_msg.wz
        wF = map_msg.wf
        wR = map_msg.wr
        epsi = map_msg.epsi
        ey = map_msg.ey
        s = map_msg.s
        self.state = np.array([vx, vy, wz, wF, wR, epsi, ey, s]).reshape((-1, 1))
        self.begin = 1

    def roll_out_trajectory(self, x_0, us, N):
        xs = np.zeros((self.n, N))
        xs[:, 0] = x_0.copy()
        for ii in range(N - 1):
            xs[:, ii + 1] = self.ar.update_dynamics(xs[:, ii], us[ii], self.dt_linearization)
        return xs

    def update_solution(self, queue):
        xs, us, D = queue.get()
        A, B, d = self.ar.linearize_dynamics(xs, us)
        A = A.reshape((self.n, self.n, self.N), order='F')
        B = B.reshape((self.n, self.m, self.N), order='F')
        d = d.reshape((self.n, 1, self.N), order='F')
        D = np.tile(D.reshape((self.n, self.l, 1)), (1, 1, self.N))
        A, B, d, D = self.ar.form_long_matrices_LTV(A, B, d, D)
        sigma_0 = np.zeros((self.n, self.n))
        sigma_N_inv = np.zeros((self.n, self.n))
        self.solver.populate_params(A, B, d, D, xs[:, 0], sigma_0, sigma_N_inv, self.Q_bar, self.R_bar, us[:, 0])
        V, K = self.solver.solve()
        K = K.reshape((self.m * self.N, self.n * self.N))
        X_bar = np.dot(A, xs[:, 0]) + np.dot(B, V) + d.flatten()
        queue.put((V, K, X_bar))

    def update_control(self, V, K, X_bar, kk):
        y = self.state.flatten() - X_bar[kk * self.n:(kk + 1) * self.n]
        u = V[kk * self.m:(kk + 1) * self.m] + np.dot(K[kk * self.m:(kk + 1) * self.m, kk * self.n:(kk + 1) * self.n], y)
        u = np.where(u > self.u_max, self.u_max, u)
        u = np.where(u < self.u_min, self.u_min, u)
        self.chassis_command.steering = u[0]
        self.chassis_command.throttle = u[1]
        self.command_pub.publish(self.chassis_command)


if __name__ == '__main__':
    controller = CS_SMPC()
    num_steps_applied = int(controller.dt_solve / controller.dt_linearization)
    solver_io = Queue(maxsize=1)
    dt_control = 0.02
    control_update_rate = rospy.Rate(1/dt_control)
    linearization_step_rate = rospy.Rate(1/controller.dt_linearization)
    us = np.array([0.0, 0.2]).reshape((-1, 1)).tile((1, controller.N))
    print('waiting for first pose estimate')
    while not controller.begin and not rospy.is_shutdown():
        rospy.sleep(0.1)
    xs = controller.roll_out_trajectory(controller.state, us, controller.N)
    D = np.zeros((controller.n, controller.l))
    solver_io.put((xs, us, D))
    solve = DummyProcess(target=controller.update_solution, args=(solver_io,))
    solve.start()
    solve.join()
    V, K, X_bar = solver_io.get()
    us = V.reshape((controller.m, controller.N), order='F')
    predicted_x_0 = controller.roll_out_trajectory(controller.state, us, num_steps_applied)
    us = np.hstack((us[:, num_steps_applied:], np.tile(us[:, -1].reshape((-1, 1)), (1, controller.N - num_steps_applied))))
    xs = controller.roll_out_trajectory(predicted_x_0, us, controller.N)
    solver_io.put((xs, us, D))
    solve = DummyProcess(target=controller.update_solution, args=(solver_io,))
    solve.start()
    for ii in range(num_steps_applied):
        for jj in range(int(controller.dt_linearization / dt_control)):
            controller.update_control(V, K, X_bar, ii)
            control_update_rate.sleep()
        # linearization_step_rate.sleep()
    V, K, X_bar = solver_io.get()
    # while not rospy.is_shutdown():






