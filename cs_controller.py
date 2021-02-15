from multiprocessing.dummy import DummyProcess, Queue
import numpy as np
import time
import matplotlib.pyplot as plt
import cs_model
import cs_solver
import rospy
from autorally_private_msgs.msg import mapCA
from autorally_msgs.msg import chassisCommand


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
        self.u_min = np.array([-0.95, 0.0])
        self.u_max = np.array([0.95, 0.95])
        self.solver = cs_solver.CSSolver(self.n, self.m, self.l, self.N, self.u_min, self.u_max)

        Q = np.zeros((self.n, self.n))
        Q[0, 0] = 1
        Q[1, 1] = 1
        Q[5, 5] = 100
        Q[6, 6] = 20
        self.Q_bar = np.kron(np.eye(self.N, dtype=int), Q)
        R = np.zeros((self.m, self.m))
        R[0, 0] = 5
        R[1, 1] = 5
        self.R_bar = np.kron(np.eye(self.N, dtype=int), R)

        self.state = np.zeros((self.n, 1))  # np.array([5, 0, 0, 50, 50, 0, 0, 0])
        self.control = np.zeros((self.m, 1))

    def map_callback(self, map_msg):
        vehicle_rF = 0.095
        vehicle_rR = 0.090
        vx = map_msg.vx
        vy = map_msg.vy
        wz = map_msg.wz
        wF = map_msg.wf / vehicle_rF
        wR = map_msg.wr / vehicle_rR
        epsi = map_msg.epsi
        ey = map_msg.ey
        s = map_msg.s
        self.state = np.array([vx, vy, wz, wF, wR, epsi, ey, s]).reshape((-1, 1))
        self.begin = 1

    def roll_out_trajectory(self, x_0, us, N, lin=None):
        if lin:
            A, B, d = lin
            xs = np.dot(A, x_0) + np.dot(B, us.flatten().reshape((-1, 1))) + d
            xs = xs.reshape((self.n, self.N), order='F')
            xs = np.hstack((x_0, xs[:, :(N-1)]))
        else:
            xs = np.zeros((self.n, N))
            xs[:, 0] = x_0.flatten()
            for ii in range(N - 1):
                xs[:, ii + 1] = self.ar.update_dynamics(xs[:, ii:ii+1], us[:, ii:ii+1], self.dt_linearization).flatten()
            # print(x_0, us, xs)
        return xs

    def update_solution(self, queue):
        x_0, us, D = queue.get()
        xs = controller.roll_out_trajectory(x_0, us, controller.N)
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
        queue.put((V, K, X_bar, (A, B, d)))

    def update_control(self, V, K, X_bar, kk):
        y = self.state.flatten() - X_bar[kk * self.n:(kk + 1) * self.n]
        u = V[kk * self.m:(kk + 1) * self.m] + np.dot(K[kk * self.m:(kk + 1) * self.m, kk * self.n:(kk + 1) * self.n], y)
        u = np.where(u > self.u_max, self.u_max, u)
        u = np.where(u < self.u_min, self.u_min, u)
        print(u)
        self.chassis_command.steering = u[0]
        self.chassis_command.throttle = u[1]
        self.command_pub.publish(self.chassis_command)
        return y


if __name__ == '__main__':
    controller = CS_SMPC()
    num_steps_applied = int(controller.dt_solve / controller.dt_linearization)
    solver_io = Queue(maxsize=1)
    dt_control = 0.02
    control_update_rate = rospy.Rate(1/dt_control)
    linearization_step_rate = rospy.Rate(1/controller.dt_linearization)
    us = np.tile(np.array([0.0, 0.2]).reshape((-1, 1)), (1, controller.N))
    print('waiting for first pose estimate')
    while not controller.begin and not rospy.is_shutdown():
        rospy.sleep(0.1)
    D = np.zeros((controller.n, controller.l))
    solver_io.put((controller.state, us, D))
    solve = DummyProcess(target=controller.update_solution, args=(solver_io,))
    solve.start()
    solve.join()
    while not rospy.is_shutdown():
        t0 = time.time()
        V, K, X_bar, lin_params = solver_io.get()
        us = V.reshape((controller.m, controller.N), order='F')
        predicted_x_0 = controller.roll_out_trajectory(controller.state, us, num_steps_applied + 1, lin=lin_params)[:, -1].reshape((-1, 1))
        print(controller.state, us, predicted_x_0)
        us = np.hstack((us[:, num_steps_applied:], np.tile(us[:, -1].reshape((-1, 1)), (1, num_steps_applied))))
        solver_io.put((predicted_x_0, us, D))
        solve = DummyProcess(target=controller.update_solution, args=(solver_io,))
        solve.start()
        print('setup time:', time.time() - t0)
        for ii in range(num_steps_applied):
            for jj in range(int(controller.dt_linearization / dt_control)):
                y = controller.update_control(V, K, X_bar, ii)
                control_update_rate.sleep()
                if ii == 1 and jj == 0:
                    D = np.diag(y)
            # linearization_step_rate.sleep()




