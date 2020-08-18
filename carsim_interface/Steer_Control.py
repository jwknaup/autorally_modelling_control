import argparse
import math
import os
import ctypes

import Simulation


if __name__ == '__main__':
    degrees_to_radians_scale_factor = 180.0 / math.pi

    parser = argparse.ArgumentParser(
        description='Python 3.5 script which runs the steering control example')

    parser.add_argument('--simfile', '-s', dest='path_to_sim_file',
                        default=os.path.join(os.getcwd(), "simfile.sim"),
                        help="Path to simfile. For example D:\\Product_dev\\Image\\CarSim\\Core\\CarSim_Data\\simfile.sim")

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
                import_array = [0.0, 0.0, 0.0]
                export_array = []

                export_array = vs.CopyExportVars(configuration.get('n_export')) # get export variables from vs solver

                # New parameters for use in the steer controller
                Lfwd = 20.0
                GainStr = 10 # This has units deg/m
                LatTrack = -1.6

                # constants to show progress bar
                print_interval = (configuration.get('t_stop') - configuration.get('t_start')) / configuration.get('t_step') / 50
                ibarg = print_interval

                status = 0

                # Check that we have enough export variables
                if (len(export_array) < 3):
                    print("At least three export parameters needed.")
                else:
                    # Run the integration loop
                    while status is 0:
                        t_current = t_current + configuration.get('t_step') # increment the time

                        # Steering Controller variables, based on previous exports
                        x_center_of_gravity = export_array[0]
                        y_center_of_gravity = export_array[1]
                        yaw = export_array[2] / degrees_to_radians_scale_factor # convert export deg to rad
                        x_preview = x_center_of_gravity + Lfwd * math.cos(yaw)
                        y_preview = y_center_of_gravity + Lfwd * math.sin(yaw)
                        road_l = vs.GetRoadL(x_preview, y_preview)

                        # copy values for 3 variables that the VS solver will import
                        import_array = [GainStr * (LatTrack - road_l), # This has units of deg
                        x_preview,
                        y_preview]

                        # Call the VS API integration function
                        status, export_array = vs.IntegrateIO(t_current, import_array, export_array)

                        # Update bar graph to show progress
                        if ibarg >= print_interval:
                            print("=", end="", flush=True)
                            ibarg = 0

                        ibarg = ibarg + 1

                # Terminate solver
                vs.TerminateRun(t_current)
