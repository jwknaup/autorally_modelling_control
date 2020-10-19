% Simple Steering control example to show how to use M-File script wrapper
% to work with a VS solver DLL using the VS API. This version uses
% vs_statement to setup the VS Solver with import and export variables to
% match the arrays defined in MATLAB.

% To run, generate files from from the CarSim browser (use the button 
% "Generate Files for this Run"), and then run this file from MATLAB.
% This is set up to look for a file named 'steer_control_vs_cmd.sim'.

% Check if last loaded library is still in memory. If so, unload it.
clc;
if libisloaded('vs_solver')
  unloadlibrary('vs_solver');
end

% Scan simfile for DLL pathname. Change the name if it's not what you use.
simfile = 'simple.sim';
SolverPath = vs_dll_path(simfile);

% Load the solver DLL
[notfound, warnings] = ...
     loadlibrary(SolverPath, 'vs_api_def_m.h', 'alias', 'vs_solver');

% libfunctionsview('vs_solver'); % uncomment to see all VS API functions

% Start and read normal inputs
t = calllib('vs_solver', 'vs_setdef_and_read', simfile, 0, 0);

% activate three export variables from VS Solver
vs_statement('EXPORT', 'Vx');
vs_statement('EXPORT', 'Vy');
vs_statement('EXPORT', 'AV_Y');
vs_statement('EXPORT', 'Ax');
vs_statement('EXPORT', 'AVy_R1');
vs_statement('EXPORT', 'AVy_R2');
vs_statement('EXPORT', 'AVy_L1');
vs_statement('EXPORT', 'AVy_L2');
vs_statement('EXPORT', 'Yaw');
vs_statement('EXPORT', 'Xcg_SM');
vs_statement('EXPORT', 'Ycg_SM');

% activate three import variables for VS Solver
vs_statement('IMPORT', 'IMP_FX_L1 REPLACE 0');
vs_statement('IMPORT', 'IMP_FX_L2 REPLACE 0');
vs_statement('IMPORT', 'IMP_FX_R1 REPLACE 0');
vs_statement('IMPORT', 'IMP_FX_R2 REPLACE 0');
vs_statement('IMPORT', 'IMP_FY_L1 REPLACE 0');
vs_statement('IMPORT', 'IMP_FY_L2 REPLACE 0');
vs_statement('IMPORT', 'IMP_FY_R1 REPLACE 0');
vs_statement('IMPORT', 'IMP_FY_R2 REPLACE 0');
vs_statement('IMPORT', 'IMP_MY_OUT_D2_L ADD 0');
vs_statement('IMPORT', 'IMP_MY_OUT_D2_R ADD 0');
vs_statement('IMPORT', 'IMP_MY_OUT_D1_L ADD 0');
vs_statement('IMPORT', 'IMP_MY_OUT_D1_R ADD 0');

calllib('vs_solver', 'vs_initialize', t, 0, 0);
disp(calllib('vs_solver', 'vs_get_output_message'));

% Define parameters that will be used in the steer controller
kph2mps = 1000/3600;
deg2rad = 3.14159 / 180;
g2mps2 = 9.81;
rpm2rps = 2*3.14159/60;

node = ros.Node('carsim');
pub = ros.Publisher(node,'/state','std_msgs/Float64MultiArray');
sub = ros.Subscriber(node,'/control','std_msgs/Float64MultiArray');

% Define import/export arrays (both with length 3) and pointers to them
imports = zeros(1, 12);
exports = zeros(1, 11);
p_imp = libpointer('doublePtr', imports);
p_exp = libpointer('doublePtr', exports);

% get time step and export variables from the initialization
t_dt = calllib('vs_solver', 'vs_get_tstep');
calllib('vs_solver', 'vs_copy_export_vars', p_exp);
stop = calllib('vs_solver', 'vs_error_occurred');
disp('The simulation is running...');

% This is the main simulation loop. Continue as long as stop is 0.
ii = 0;
while ~stop 
    t = t + t_dt;  % increment time

    % Update the array of exports using the pointer p_exp
    exports = get(p_exp, 'Value');
    state_msg = rosmessage('std_msgs/Float64MultiArray');
    vx = exports(1) * kph2mps;
    vy = exports(2) * kph2mps;
    wz = exports(3) * deg2rad;
    ax = exports(4) * g2mps2;
    wF = (exports(5) * rpm2rps + exports(7) * rpm2rps) / 2;
    wR = (exports(6) * rpm2rps + exports(8) * rpm2rps) / 2;
    yaw = exports(9) * deg2rad;
    X = exports(10);
    Y = exports(11);
    
%     state_msg.vx = vx;
%     state_msg.vy = vy;
%     state_msg.wz = wz;
%     state_msg.wf = wF;
%     state_msg.wr = wR;
%     state_msg.s = ax;
    if mod(t, 0.01) < 0.001
        state_msg.Data = [vx, vy, wz, wF, wR, yaw, X, Y];
        send(pub, state_msg)
%         ii = ii + 1
%         t
        ctrl_msg = receive(sub);
        steering = ctrl_msg.Data(1);
        command_wR = ctrl_msg.Data(2);
    end
    input_tensor = [steering, vx, vy, wz, ax, wF, wR];
    forces = update_dynamics(input_tensor);
    fafy = forces(1);
    fary = forces(2);
    fafx = forces(3);
    farx = forces(4);
    torque = 1 * (command_wR - wR);
    imports = [fafx/2, farx/2, fafx/2, farx/2, fafy/2, fary/2, fafy/2, fary/2, torque, torque, 0, 0];

  % copy values into import array and set pointer for the VS solver
%   imports(1)= GainStr*(LatTrack - road_l); 
%   imports(2)= Xpreview;
%   imports(3)= Ypreview;
%     imports(9) = 10;
%     imports(10) = 10;
    set(p_imp, 'Value', imports); %set pointer for array of imports

    % Call VS API integration function and exchange import and export arrays
    stop = calllib('vs_solver', 'vs_integrate_io', t, p_imp, p_exp);
end

% Terminate solver
calllib('vs_solver', 'vs_terminate_run', t);
disp('The simulation has finished.');

% Unload solver DLL
unloadlibrary('vs_solver');
clear('sub', 'pub', 'node')

function forces = update_dynamics(input)
m_Vehicle_m = 21.7562;%1270;%21.7562;
m_Vehicle_Iz = 1.124;%2000;%2.5;
lFR = 0.57;%2.91;%0.57;
m_Vehicle_lF = 0.34;%1.022;%0.4;
m_Vehicle_lR = lFR-m_Vehicle_lF;%0.57
m_Vehicle_IwF = 0.01;%8.0;%4.02;
m_Vehicle_IwR = 0.5;%3.73;
m_Vehicle_rF = 0.095;%0.325;%0.095;
m_Vehicle_rR = 0.09;%0.325;%0.090;
m_Vehicle_mu1 = 0.75;
m_Vehicle_mu2 = 0.90;
m_Vehicle_h = 0.12;%0.54;%.2;%0.2;    
m_g = 9.80665;

m_Vehicle_kSteering = 18.7861;%23.0811;%34
m_Vehicle_cSteering = 0.0109;
m_Vehicle_kThrottle = 165.0922;
m_Vehicle_kTorque = 0.07577;

tire_B = 4;%1.5;
tire_C = 1;%1.5;
tire_D = 1.0;
tire_a = 1;%0.75;
tire_E = 0.97;
tire_Sh = -0.0;
tire_Sv = 0.00;
tire_By = tire_B;
tire_Cy = tire_C;
tire_Dy = tire_C;
tire_Bx = tire_B;
tire_Cx = tire_C;
tire_Dx = tire_D;

m_nx = 8;
m_nu = 2;
% N = size(state,2);

steering = input(1);
vx = input(2);
vy = input(3);
wz = input(4);
ax = input(5);
wF = input(6);
wR = input(7); 

delta = m_Vehicle_kSteering * steering + m_Vehicle_cSteering;
% delta = (input(2) + input(3))/2;
% delta = input(1);
% T = m_Vehicle_kThrottle * input(2, 1);

min_velo = 0.1;
deltaT = 0.01;

% if (vx < min_velo)
%   vx = min_velo;
% end
% if (wF < min_velo / m_Vehicle_rF)
%     wF = min_velo / m_Vehicle_rF + 1e-4;
% end
% if wR < min_velo / m_Vehicle_rR
%     wR = min_velo / m_Vehicle_rR + 1e-4;
% end
% if (vx ~= 0.0)
beta = atan2(vy, vx);
% else
%   beta = 0.0;
% end

V = sqrt(vx .* vx + vy .* vy);
vFx = V .* cos(beta - delta) + wz .* m_Vehicle_lF .* sin(delta);
vFy = V .* sin(beta - delta) + wz .* m_Vehicle_lF .* cos(delta);
vRx = vx;
vRy = vy - wz .* m_Vehicle_lR;

alphaF = -atan(vFy ./ abs(vFx));
alphaR = -atan(vRy ./ abs(vRx));

% if (wF ~= 0.0)
  sFx = (vFx - wF .* m_Vehicle_rF) ./ (wF .* m_Vehicle_rF);
% else
%   sFx = 0.0;
% end
% if (wR ~= 0.0)
  sRx = (vRx - wR .* m_Vehicle_rR) ./ (wR .* m_Vehicle_rR);
% else
%   sRx = 0.0;
% end
% if (vFx ~= 0.0)
  sFy = vFy ./ (wF .* m_Vehicle_rF);
%   sFy = (1 + sFx) * vFy / vFx;
% else
%   sFy = 0.0;
% % end
% if (vRx ~= 0.0)
  sRy = vRy ./ (wR .* m_Vehicle_rR);
%   sRy = (1 + sRx) * vRy / vRx;
% else
%   sRy = 0.0;
% end

sF = sqrt(sFx .* sFx + sFy .* sFy);
sR = sqrt(sRx .* sRx + sRy .* sRy);

sEF = sF - tire_Sh;
sER = sR - tire_Sh;

% muF = tire_D*sin( tire_C*atan( tire_B*sEF - tire_E*(tire_B*sEF - atan(tire_B*sEF) ) ) ) + tire_Sv;
% muR = tire_D*sin( tire_C*atan( tire_B*sER - tire_E*(tire_B*sER - atan(tire_B*sER) ) ) ) + tire_Sv;
muF = tire_D .* sin(tire_C .* atan(tire_B .* sF)); 
muR = tire_D .* sin(tire_C .* atan(tire_B .* sR));

muFx = -sFx ./ sF .* muF;
muFy = -sFy ./ sF .* muF;
muRx = -sRx ./ sR .* muR;
muRy = -sRy ./ sR .* muR;

fFz = m_Vehicle_m .* m_g .* (m_Vehicle_lR - m_Vehicle_h .* muRx) ./ (m_Vehicle_lF + m_Vehicle_lR + m_Vehicle_h .* (muFx .* cos(delta) - muFy .* sin(delta) - muRx));
% fFz = m_Vehicle_m * m_g * (m_Vehicle_lR / lFR);
fRz = m_Vehicle_m .* m_g - fFz;

fFwx = muFx .* fFz;
fFwy = tire_a.*muFy .* fFz;
fRx = muRx .* fRz;
fRy = tire_a.*muRy .* fRz;

fFy = fFwy * cos(delta) + fFwx * sin(delta);
fFx = fFwx * cos(delta) - fFwy * sin(delta);

forces = [fFy, fRy, fFx, fRx];

%%%%%%%%%%%%%%%%%%%
% sEF = -(vFx - wF * m_Vehicle_rF) / (vFx) + tire_Sh;
% muFx = tire_Dx*sin( tire_Cx*atan( tire_Bx*sEF - tire_E*(tire_Bx*sEF - atan(tire_Bx*sEF) ) ) ) + tire_Sv;
% sEF = -(vRx - wR * m_Vehicle_rR) / (vRx) + tire_Sh;
% muRx = tire_Dx*sin( tire_Cx*atan( tire_Bx*sEF - tire_E*(tire_Bx*sEF - atan(tire_Bx*sEF) ) ) ) + tire_Sv;
% 
% sEF = atan(vFy / abs(vFx)) + tire_Sh;
% muFy = -tire_Dy*sin( tire_Cy*atan( tire_By*sEF - tire_E*(tire_By*sEF - atan(tire_By*sEF) ) ) ) + tire_Sv;
% sEF = atan(vRy / abs(vRx)) + tire_Sh;
% muRy = -tire_Dy*sin( tire_Cy*atan( tire_By*sEF - tire_E*(tire_By*sEF - atan(tire_By*sEF) ) ) ) + tire_Sv;

% fFz = m_Vehicle_m * m_g * (m_Vehicle_lR - m_Vehicle_h * muRx) / (m_Vehicle_lF + m_Vehicle_lR + m_Vehicle_h * (muFx * cos(delta) - muFy * sin(delta) - muRx));
% fFz = m_Vehicle_m * m_g * (m_Vehicle_lR / lFR);
% fRz = m_Vehicle_m * m_g - fFz;

% fFx = fFz * muFx;
% fRx = fRz * muRx;
% fFy = fFz * muFy;
% fRy = fRz * muRy;

% fFx = input(6);
% fFy = input(7);
% fRx = input(8);
% fRy = input(9);
%%%%%%%%%%%%%%%%%%%

% dot_X = vx .* cos(Yaw) - vy .* sin(Yaw);
% dot_Y = vx .* sin(Yaw) + vy .* cos(Yaw); 
% 
% next_state = zeros(m_nx, N);
% next_state(1, :) = vx + deltaT .* ((fFx .* cos(delta) - fFy .* sin(delta) + fRx) ./ m_Vehicle_m + vy .* wz);
% next_state(2, :) = vy + deltaT .* ((fFx .* sin(delta) + fFy .* cos(delta) + fRy) ./ m_Vehicle_m - vx .* wz);
% next_state(3, :) = wz + deltaT .* ((fFy.*cos(delta) + fFx.*sin(delta)) .* m_Vehicle_lF - fRy .* m_Vehicle_lR) ./ m_Vehicle_Iz;
% 
% % fafy = input(7,1);
% % fary = input(8,1);
% % % next_state(2, 1) = vy + deltaT * ((fafy + fary) / m_Vehicle_m - vx * wz);
% % next_state(3, 1) = wz + deltaT * ((fafy) * m_Vehicle_lF - fary * m_Vehicle_lR) / m_Vehicle_Iz;
% % wz_dot = input(9,:);
% % next_state(3, 1) = wz + deltaT * wz_dot;
% 
% next_state(4, :) = wF - deltaT .* m_Vehicle_rF ./ m_Vehicle_IwF .* fFx;
% next_state(5, :) = wR;
% next_state(6, :) = Yaw + deltaT .* (wz);
% next_state(7, :) = X + deltaT .* dot_X;
% next_state(8, :) = Y + deltaT .* dot_Y;

% state = next_state;
end