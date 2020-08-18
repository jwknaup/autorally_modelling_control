import argparse

import os
import struct
import sys
# Short routine to demonstrate operation of Python interface to the VS API using SWIG
if __name__ == '__main__':
    if sys.version_info[0] < 3:
        print("Python version must be greater than 3.0")
    else:
        systemSize = (8 * struct.calcsize("P"))  # 32 or 64
        #Import the interface
        if (systemSize == 32):
            from Python_32bit import pyvs
        else:
            from Python_64bit import pyvs

        parser = argparse.ArgumentParser(
            description='Python 3.5 script that runs the simulation in simfile.sim in the current directory.')

        parser.add_argument('--simfile', '-s', dest='path_to_sim_file',
                        default=os.path.join(os.getcwd(), "simfile.sim"),
                        help="Path to simfile. For example C:\\Programs\\CarSim\\CarSim_Data\\simfile.sim")

        parser.add_argument('--runs', '-r', type=int, dest='number_of_runs',
                        default=1,
                        help="Number of runs to make per single load of DLL. This parameter exists to replicate how real-time system use the solver")

        args = parser.parse_args()
        simfile = args.path_to_sim_file
        number_of_runs = args.number_of_runs
        if number_of_runs < 1:
            number_of_runs = 1

        #Set a sample simfile
        print("Attempting to load:\n\n\t ",simfile)

        #Get the path to the solver from the simfile
        ret1, dllPath = pyvs.vs_get_dll_path(simfile)
        if "Default64" in dllPath:
            dllSize = 64
        else:
            dllSize = 32

        if systemSize != dllSize:
            print("Python size (32/64) must match size of .dlls being loaded.")
            print("Python size:", systemSize,"DLL size:",dllSize)
        else:   #systems match, we can continue
            #Load the library
            dllModule = pyvs.vs_load_library(dllPath)

            #Load the API
            ret2 = pyvs.vs_get_api(dllModule, dllPath)

            #Run solver solver with simfile
            for i in range(0, number_of_runs):
                print(os.linesep + "++++++++++++++++ Starting run number: " + str(i + 1) + " ++++++++++++++++" + os.linesep)
                ret3 = pyvs.vs_run(simfile)
                if ret3 is not 0:
                    break
                print(os.linesep + "++++++++++++++++ Ending run number: " + str(i + 1) + " ++++++++++++++++" + os.linesep)

            #Get echoFile name and version of model
            echoFile = pyvs.vs_get_echofile_name()
            print(echoFile)

            versionModel = pyvs.vs_get_version_model()
            print(versionModel)

            #Free the module
            pyvs.vs_free_library(dllModule)
