import argparse
import math
import os
import ctypes
import numpy as np
import torch

from carsim_interface import Simulation
import hybrid_nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Python 3.5 script which runs the steering control example')

    parser.add_argument('--simfile', '-s', dest='path_to_sim_file',
                        default=os.path.join(os.getcwd(), "carsim_interface", "simfile.sim"),
                        help="Path to simfile. For example D:\\Product_dev\\Image\\CarSim\\Core\\CarSim_Data\\simfile.sim")

    kph2mps = 1000/3600
    deg2rad = 3.14159 / 180
    g2mps2 = 9.81
    rpm2rps = 2*3.14159/60

    args = parser.parse_args()
    sim_file_filename = args.path_to_sim_file
    vs = Simulation.VehicleSimulation()
    path_to_vs_dll = vs.get_dll_path(sim_file_filename)
    if path_to_vs_dll is not None and os.path.exists(path_to_vs_dll):
        vs_dll = ctypes.cdll.LoadLibrary(path_to_vs_dll)
        if vs_dll is not None:
            if vs.get_api(vs_dll):
                # Call vs_read_configuration to read parsfile and initialize the VS solver.
                # Get no. of import/export variables, start time, stop time and time step.
                configuration = vs.ReadConfiguration(sim_file_filename)
                t_current = configuration.get('t_start')

                # Create import and export arrays based on sizes from VS solver
                export_array = vs.CopyExportVars(configuration.get('n_export')) # get export variables from vs solver

                # constants to show progress bar
                print_interval = (configuration.get('t_stop') - configuration.get('t_start')) / configuration.get('t_step') / 50
                ibarg = print_interval

                status = 0

                # Check that we have enough export variables
                if (len(export_array) < 0):
                    print("At least three export parameters needed.")
                else:
                    # Run the integration loop
                    model = hybrid_nn.Net()
                    model.load_state_dict(torch.load('hybrid_net_ar2.pth'))
                    while status is 0:
                        t_current = t_current + configuration.get('t_step') # increment the time

                        steering = 0.01
                        command_wR = 20

                        vx = export_array[0] * kph2mps
                        vy = export_array[1] * kph2mps
                        wz = export_array[2] * deg2rad
                        ax = export_array[3] * g2mps2
                        wF = export_array[4] * rpm2rps
                        wR = export_array[5] * rpm2rps

                        input_tensor = torch.from_numpy(np.vstack((steering, vx, vy, wz, ax, wF, wR)).T).float()
                        forces = model(input_tensor).detach().numpy()
                        fafy = forces[:, 0]
                        fary = forces[:, 1]
                        fafx = forces[0, 2]
                        farx = forces[0, 3]
                        torque = 1 * (command_wR - wR)
                        import_array = [fafx/2, farx/2, fafx/2, farx/2, fafy/2, fary/2, fafy/2, fary/2, torque, torque]

                        # Call the VS API integration function
                        status, export_array = vs.IntegrateIO(t_current, import_array, export_array)

                        # Update bar graph to show progress
                        if ibarg >= print_interval:
                            print("=", end="", flush=True)
                            ibarg = 0

                        ibarg = ibarg + 1

                # Terminate solver
                vs.TerminateRun(t_current)
