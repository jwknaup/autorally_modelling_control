import argparse
from ctypes import cdll
import os
import struct

import sys

import Simulation

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        error_occurred = "Python version must be 3.0 or greater."
    else:
        systemSize = (8 * struct.calcsize("P"))  # 32 or 64

        parser = argparse.ArgumentParser(
            description='Python 3.5 script that runs the simulation in simfile.sim in the current directory.')

        parser.add_argument('--simfile', '-s', dest='path_to_sim_file',
                        default=os.path.join(os.getcwd(), "simfile.sim"),
                        help="Path to simfile. For example D:\\Product_dev\\Image\\CarSim\\Core\\CarSim_Data\\simfile.sim")

        parser.add_argument('--runs', '-r', type=int, dest='number_of_runs',
                        default=1,
                        help="Number of runs to make per single load of DLL. This parameter exists to replicate how real-time system use the solver")

        args = parser.parse_args()
        path_to_sim_file = args.path_to_sim_file
        number_of_runs = args.number_of_runs
        if number_of_runs < 1:
            number_of_runs = 1

        vs = Simulation.VehicleSimulation()
        path_to_vs_dll = vs.get_dll_path(path_to_sim_file)
        error_occurred = 1
        if path_to_vs_dll is not None and os.path.exists(path_to_vs_dll):
            if "Default64" in path_to_vs_dll:
                dllSize = 64
            else:
                dllSize = 32
            
            if systemSize != dllSize:
                print("Python size (32/64) must match size of .dlls being loaded.")
                print("Python size:", systemSize, "DLL size:", dllSize)
            else:  # systems match, we can continue
                vs_dll = cdll.LoadLibrary(path_to_vs_dll)
                if vs_dll is not None:
                    if vs.get_api(vs_dll):
                        for i in range(0, number_of_runs):
                            print(os.linesep + "++++++++++++++++ Starting run number: " + str(i + 1) + " ++++++++++++++++" + os.linesep)
                            error_occurred = vs.run(path_to_sim_file.replace('\\\\', '\\'))
                            if error_occurred is not 0:
                                break
                            print(os.linesep + "++++++++++++++++ Ending run number: " + str(i + 1) + " ++++++++++++++++" + os.linesep)

    sys.exit(error_occurred)
