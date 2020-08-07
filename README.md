# Vehicle Dynamics Model

## Data preparation
`load_bag_single_track_cartesian.m` will load a rosbag file and preprocess the data to be in the expected format for evaluation and system ID. However, this process can be skipped because the formatted data from the mppi dataset is already stored in the directory `mppi_data/`. Therefore, simply select and load one of the `mppi_states_controls#.mat` files into the workspace.

## Adaptive Unscented Kalman Filter
`Single_Track_Cartesian_AJUKF.m` contains the Kalman filter code for identifying the unknown system parameters. After loading on of the `mppi_states_controls#.mat` files it can be run with `[ para_opt, X_train, Y_opt, Y_train ] = Single_Track_Cartesian_AJUKF(states, inputs, 0.01);`. It will run over the training data, plot the results and output the optimized parameters. 

## Neural Networks
`wz_nn.py` will train and evaluate a neural net to model the steering angle control. Archived neural net training code (eg for adding a bias term to the state equation(s) is in `dyn_error_nn.py`). `ff_nn.py` and `hybrid_nn.py` are the latest neural network training codes for training and validating a friction model or friction/steering error model respectively.

## Evaluation
`single_track_data_gen_cartesian.m` will run the bicycle model with the current states and controls and plot the results. 

## Controller
`mpc_testing.m` contains matlab implementations of an LTI and LTV MPC and tests them 

## References
`acc17b.pdf` is the reference paper for the Kalman filter code. The code closely follows `Algorithm I` in the paper. `model.pdf` provides relevant equations for the bicycle model for reference. `falcone.pdf` is the reference paper for the LTI and LTV MPC implementations (specifically Section III).
