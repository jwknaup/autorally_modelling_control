%%
% dddd = [0.1, 0.01, 0.1, 1, 1, 2, 2, 2, 0.001, 1];
% [A, B, d] = linearize_dynamics(state, control, dddd)
[epsi, ey, s] = map_tf(0,0,0)
%%
clc
% master = ros.Core;
% node = ros.Node('mpc');
% pub = ros.Publisher(node,'/control','std_msgs/Float64MultiArray');
% sub = ros.Subscriber(node,'/state','std_msgs/Float64MultiArray');

N = 2000;
Nf = 2000;
% ddd = [0.1, 0.01, 0.1, 10, 10, 2, 2, 2, 0.001, 10];
ddd = [0.1, 0.01, 0.1, 10, 10, 0.5, 1, 2, 0.001, 10];
ddd = ones(1, 10) .* 0.001;
tend = 12;
subset_states = states(:,N:10:end-Nf);
subset_inputs = [inputs(1,N:10:end-Nf); states(5,N:10:end-Nf)];
state = subset_states(:,1);
yaw_new = mod(state(6), 2*pi);
if yaw_new > pi
    yaw_new = yaw_new - 2*pi;
end
yaw_offset = yaw_new - state(6);
subset_states(6, :) = subset_states(6, :) + yaw_offset;
state = subset_states(:,1);
state = [2; 0; 0; 20; 20; -0.69; -8.7; 4.5];

%%
% state(4) = state(5);
control = subset_inputs(:,1);
control = [0; 0]
% time = (1:10000)/100;
time = (1:length(subset_states))/100;
sim_states = zeros(length(state)+3, length(time));
sim_controls = zeros(length(control), length(time));
% x_target = subset_states(:,end);
% x_target = subset_states(:,tend);
x_target = [3; 0; 0; 0; 0; 0; 0; 0];
[epsi, ey, s, curvature] = map_tf(state(6), state(7), state(8));
state_track = [state(1:5); epsi; ey; s]
[control, xs, us] = LTIMPC(state_track, control, x_target, ddd, curvature);
state = update_dynamics(state, control)
% state = sim(control, pub, sub)
[epsi, ey, s, curvature] = map_tf(state(6), state(7), state(8));
state_track = [state(1:5); epsi; ey; s];
xs{1} = state_track;
control
for ii = 1:length(time)-tend
%     x_target = subset_states(:,ii+tend);
%     x_target(1) = 5;
%     [control, xs, us] = LTIMPC(xs{1}, control, x_target, ddd);
    [control, xs, us] = LTVMPC(xs, us, x_target, ddd, curvature)
    if isnan(control)
        break
    end
%     control = subset_inputs(:, ii);
%     ii
%     state = update_dynamics(state, control);
%     state = sim(control, pub, sub);
    [epsi, ey, s, curvature] = map_tf(state(6), state(7), state(8));
    state_track = [state(1:5); epsi; ey; s];
    xs{1} = state_track;
    sim_states(1:8,ii) = state;
    sim_states(9:11,ii) = state_track(6:8);
    sim_controls(:,ii) = control;
end
clear('sub', 'pub', 'node')
clear('master')
%%
plot_subset = subset_states([1,2,3,6,7,8], :);
plot_sim = sim_states([1,2,3,6,7,8,9,10,11], :);
state = plot_subset(:,1);
figure;
for ii = 1:12
    subplot(12/2, 2, ii);
    hold on
    if ii < 10
%         plot(time(:,1:end-tend), plot_subset(ii, 1:end-tend))
        plot(time(:,1:end-tend), plot_sim(ii, 1:end-tend))
    elseif ii == 10
        plot(sim_states(7, 1:end-tend), sim_states(8, 1:end-tend));
    else
%         plot(time(:,1:end-tend), subset_inputs(ii-6, 1:end-tend));
        plot(time(:,1:end-tend), sim_controls(ii-10, 1:end-tend));
    end
end
%%
state = [1,2,3,4,5,6,7,8,9,10,11,12];
subplot(length(state)/2,2,1);
legend('vx');
xlabel('t (s)');
ylabel('m/s')
subplot(length(state)/2,2,2);
legend('vy');
xlabel('t (s)');
ylabel('m/s');
subplot(length(state)/2,2,3);
legend('yaw-rate');
xlabel('t (s)');
ylabel('rad/s');
subplot(length(state)/2,2,4);
legend('Yaw');
xlabel('t (s)');
ylabel('rad');
subplot(length(state)/2,2,5);
legend('X');
xlabel('t (s)');
ylabel('m');
subplot(length(state)/2,2,6);
legend('Y');
xlabel('t (s)');
ylabel('m');
subplot(length(state)/2,2,7);
legend('yaw error');
xlabel('t (s)');
ylabel('rad');
subplot(length(state)/2,2,8);
legend('lateral error');
xlabel('t (s)');
ylabel('m');
subplot(length(state)/2,2,9);
legend('position along track');
xlabel('t (s)');
ylabel('m');
subplot(length(state)/2,2,10);
legend('path');
xlabel('X (m)');
ylabel('Y (m)');
subplot(length(state)/2,2,11);
legend('steering');
xlabel('t (s)');
% ylabel('m');
subplot(length(state)/2,2,12);
legend('wheel speed');
xlabel('t (s)');
ylabel('rad/s');
% legend('Goal vx','Realized vx');
% xlabel('t (s)');
% ylabel('m/s')
% subplot(length(state)/2,2,2);
% legend('Recorded vy','Realized vy');
% xlabel('t (s)');
% ylabel('m/s');
% subplot(length(state)/2,2,3);
% legend('Recorded yaw-rate','Realized yaw-rate');
% xlabel('t (s)');
% ylabel('rad/s');
% % subplot(4,2,4);
% % legend('Goal wF','Simulated wF','Carsim wF');
% % xlabel('t (s)');
% % ylabel('rad/s');
% % subplot(4,2,5);
% % legend('True wR','Simulated wR','Carsim wR');
% % xlabel('t (s)');
% % ylabel('rad/s');
% subplot(length(state)/2,2,4);
% legend('Goal Yaw','Realized Yaw');
% xlabel('t (s)');
% ylabel('rad');
% subplot(length(state)/2,2,5);
% legend('Goal X','Realized X');
% xlabel('t (s)');
% ylabel('m');
% subplot(length(state)/2,2,6);
% legend('Goal Y','Realized Y');
% xlabel('t (s)');
% ylabel('m');
% subplot(length(state)/2,2,7);
% legend('Recorded Steering','Commanded Steering');
% xlabel('t (s)');
% % ylabel();
% subplot(length(state)/2,2,8);
% legend('Recorded wheel-speed','Commanded wheel-speed');
% xlabel('t (s)');
% ylabel('rad/s');
figure;
plot(track_inner(:,1), track_inner(:,2), 'k');
hold on;
plot(track_outer(:,1), track_outer(:,2), 'k', 'HandleVisibility','off');
plot(sim_states(7, 1:end-tend), sim_states(8, 1:end-tend), 'b');
legend('track boundaries', 'path');
xlabel('X (m)');
ylabel('Y (m)');
% test_models(states, inputs);

function [control, xs, us] = LTVMPC(xs, us, x_target, ddd, curvature)
    Q = zeros(8);
    Q(1,1) = 10;
%     Q(2,2) = 0.01;
%     Q(3,3) = 10;
    Q(6,6) = 10;
    Q(7,7) = 10;
%     Q(8,8) = .01;
    QT = zeros(8);
    QT(1,1) = 1000;
    QT(6,6) = 1000;
    QT(7,7) = 1000;
    R = eye(2);
    R(1,1) = 100;
    R(2,2) = 0.0001;
    [A_0, B_0, d_0] = linearize_dynamics(xs{1}, us{1}, ddd, curvature);
    A = cell(12,1);
    B = cell(12,1);
    d = cell(12,1);
    for ii = 2:12
        [A{ii-1}, B{ii-1}, d{ii-1}] = linearize_dynamics(xs{ii}, us{ii}, ddd, curvature);
    end
    [A{12}, B{12}, d{12}] = linearize_dynamics(xs{13}, us{12}, ddd, curvature);
    params.A_0 = A_0;
%     A = zeros(8);
%     params.A = {A_0,A,A,A,A,A,A,A,A,A,A,A};
    params.A = A;
    params.B_0 = B_0;
%     B = zeros([8,2]);
%     B = B_0;
%     params.B = {B,B,B,B,B,B,B,B,B,B,B,B};
    params.B = B;
    params.Q = Q;
    params.QT = QT;
    params.R = R;
    params.d_0 = d_0;
%     d = zeros(8,1);
%     d = d_0;
%     params.d = {d,d,d,d,d,d,d,d,d,d,d,d};
    params.d = d;
%     params.half_road_width = 900;
    params.S = [0.002; 1];
    params.target = x_target;
    params.umax = [0.015; 40];
    params.x_0 = xs{1};
    params.w = 1.8;
    settings.verbose = 0;
%     settings.max_iters = 100;
%     settings.eps = 1;
%     settings.resid_tol = 1e-2;
    [vars, status] = csolve(params, settings);
    if status.converged
        control = vars.u_0;
        us = vars.u;
        xs = vars.x;
        if control(2) < 0
            control(2) = 0;
        end
    else
        xs{:}
        us{:}
        control = nan;
        us = nan;
        xs = nan;
%         'nonconvered'
    end
end

function [control, xs, us] = LTIMPC(x_0, u_0, x_target, ddd, curvature)
    Q = zeros(8);
    Q(1,1) = 10;
%     Q(2,2) = 1;
%     Q(3,3) = 0.1;
    Q(6,6) = 10;
    Q(7,7) = 10;
%     Q(8,8) = 1;
    QT = zeros(8);
    QT(1,1) = 1000;
    QT(6,6) = 1000;
    QT(7,7) = 1000;
    R = zeros(2);
    R(1,1) = 100;
    R(2,2) = 0.01;
%     x_target = [5;0;0;50;50;0;0;0];
    % x_0 = [3;0;0;30;30;0;0;0];
    % u_0 = [0; 30];
    [A, B, d] = linearize_dynamics(x_0, u_0, ddd, curvature)
    params.A_0 = A;
    A0 = eye(8);
    A0 = A;
    params.A = {A0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0};
    params.B_0 = B;
    B0 = zeros([8,2]);
    B0 = B;
    params.B = {B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0};
    params.Q = Q;
    params.QT = QT;
    params.R = R;
    params.d_0 = d;
    d0 = zeros([8,1]);
    d0 = d;
    params.d = {d0,d0,d0,d0,d0,d0,d0,d0,d0,d0,d0,d0};
%     params.half_road_width = 900;
    params.S = [0.001; 0.01];
    params.target = x_target;
    params.umax = [0.015; 40];
    params.x_0 = x_0;
    params.w = 2.4;
    settings.verbose = 0;
%     settings.max_iters = 100;
    settings.eps = 1;
    settings.resid_tol = 1e-2;
    [vars, status] = csolve(params, settings);
    if status.converged
        control = vars.u_0;
        xs = vars.x;
        us = vars.u;
        if control(2) < 0
            control(2) = 0;
        end
    else
        control = nan;
        us = vars.u
        xs = vars.x
%         'nonconvered'
    end
end

function test_models(states, inputs)
    N = 2000;
    Nf = 2000;
    ddd = 0.000001;
    subset_states = states(:,N:10:end-Nf);
    subset_inputs = [inputs(1,N:10:end-Nf); states(5,N:10:end-Nf)];
    state = subset_states(:,1);
    time = (1:length(subset_states))/100;
    nonlinear_states = zeros(length(state), length(time));
    linear_states = zeros(length(state), length(time));
    non_state = state;
    lin_state = state;

    for ii = 1:length(time)
    non_state = update_dynamics(non_state, subset_inputs(:,ii));
    nonlinear_states(:,ii) = non_state;
    [A, B, d] = linearize_dynamics(lin_state, subset_inputs(:,ii), ddd);
    lin_state = A*lin_state + B*subset_inputs(:,ii) + d;
    linear_states(:,ii) = lin_state;
    end

    figure;
    for ii = 1:length(state)
        subplot(length(state)/2, 2, ii);
        hold on
        plot(time, subset_states(ii, :))
        plot(time, nonlinear_states(ii, :))
        plot(time, linear_states(ii, :))
    end
end

function [A, B, d] = linearize_dynamics(state, control, delta, curvature)
dx = length(state);
du = length(control);

x = repmat(state, [1,dx]);
u = repmat(control, [1,dx]);
% delta_x = eye(dx) * delta;
delta_x = diag(delta(1:dx));
x_plus = x + delta_x;
x_minus = x - delta_x;
f_plus = update_dynamics(x_plus, u, curvature);
f_minus = update_dynamics(x_minus, u, curvature);
A = (f_plus - f_minus) ./ (2*delta(1:dx));

x = repmat(state, [1,du]);
u = repmat(control, [1,du]);
% delta_u = eye(du) * delta;
delta_u = diag(delta(dx+1:end));
u_plus = u + delta_u;
u_minus = u - delta_u;
f_plus = update_dynamics(x, u_plus, curvature);
f_minus = update_dynamics(x, u_minus, curvature);
B = (f_plus - f_minus) ./ (2*delta(dx+1:end));

next_state = update_dynamics(state, control, curvature);
d = next_state - A*state - B*control;

A = A(1:8,1:8);
B = B(1:8,:);
d = d(1:8,:);
end

function [next_state, curvature] = update_dynamics(state, control, varargin)
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
m_Vehicle_kThrottle = 1;%165.0922;
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
N = size(state,2);

vx = state(1, :);
vy = state(2, :);
wz = state(3, :);
wF = state(4, :);
wR = state(5, :);
if isempty(varargin)
    Yaw = state(6, :);
    X = state(7, :);
    Y = state(8, :);
else
    epsi = state(6, :);
    ey = state(7, :);
    s = state(8, :);
end

delta = m_Vehicle_kSteering * control(1, :) + m_Vehicle_cSteering;
% delta = (input(2) + input(3))/2;
% delta = input(1);
T = m_Vehicle_kThrottle * control(2, :);

min_velo = 0.1;
deltaT = 0.01;

if any(vx < min_velo)
  vx = ones(size(vx)) .* min_velo;
end
if any(wF < min_velo / m_Vehicle_rF)
    wF = ones(size(wF)) .* min_velo / m_Vehicle_rF + 1e-4;
end
if any(wR < min_velo / m_Vehicle_rR)
    wR = ones(size(wR)) .* min_velo / m_Vehicle_rR + 1e-4;
end
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

fFx = muFx .* fFz;
fFy = tire_a.*muFy .* fFz;
fRx = muRx .* fRz;
fRy = tire_a.*muRy .* fRz;

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

next_state = zeros(m_nx, N);
next_state(1, :) = vx + deltaT .* ((fFx .* cos(delta) - fFy .* sin(delta) + fRx) ./ m_Vehicle_m + vy .* wz);
next_state(2, :) = vy + deltaT .* ((fFx .* sin(delta) + fFy .* cos(delta) + fRy) ./ m_Vehicle_m - vx .* wz);
next_state(3, :) = wz + deltaT .* ((fFy.*cos(delta) + fFx.*sin(delta)) .* m_Vehicle_lF - fRy .* m_Vehicle_lR) ./ m_Vehicle_Iz;

% fafy = input(7,1);
% fary = input(8,1);
% % next_state(2, 1) = vy + deltaT * ((fafy + fary) / m_Vehicle_m - vx * wz);
% next_state(3, 1) = wz + deltaT * ((fafy) * m_Vehicle_lF - fary * m_Vehicle_lR) / m_Vehicle_Iz;
% wz_dot = input(9,:);
% next_state(3, 1) = wz + deltaT * wz_dot;

next_state(4, :) = wF - deltaT .* m_Vehicle_rF ./ m_Vehicle_IwF .* fFx;
next_state(5, :) = T;
if isempty(varargin)
    dot_X = vx .* cos(Yaw) - vy .* sin(Yaw);
    dot_Y = vx .* sin(Yaw) + vy .* cos(Yaw); 
    next_state(6, :) = Yaw + deltaT .* (wz);
    next_state(7, :) = X + deltaT .* dot_X;
    next_state(8, :) = Y + deltaT .* dot_Y;
else
    curvature = varargin{1};
    next_state(6, :) = epsi + deltaT .* (wz - (vx .* cos(epsi) - vy .* sin(epsi)) / (1 - curvature .* ey) .* curvature);
    next_state(7, :) = ey + deltaT .* (vx .* sin(epsi) + vy .* cos(epsi));
    next_state(8, :) = s + deltaT .* (vx .* cos(epsi) - vy .* sin(epsi)) / (1 - curvature .* ey);
end

% state = next_state;
end

function state = sim(control, pub, sub)
    ctrl_msg = rosmessage('std_msgs/Float64MultiArray');
    ctrl_msg.Data = control;
    send(pub, ctrl_msg)
    state_msg = receive(sub);
    state = state_msg.Data;
end

function [epsi, ey, s, curvature_] = map_tf(yaw, x, y)
    map_flag_ = 2;
    map_parameters_ = ...
	 	{
	 		{
				{-8.62, 8.38, 2.36, 0, 5.9821, 0},					 
				{-16.94, -0.19, -0.80, 5.9821, 18.7621, 0.1674}, 
				{-8.81, -8.64, -0.80, 24.7552, 11.726, 0}, 
				{-0.12, 0.05, 2.36, 36.4702, 19.304, 0.1627}, 
				{-4.37, 4.17, 2.36, 55.774, 5.919, 0}
			},

			{
				{2.78,-2.97,-0.6613, 0, 3.8022, 0},
				{10.04,6.19, 2.4829, 3.8022, 18.3537,0.1712 }, 
				{1.46, 13.11,2.4829, 22.1559, 11.0228 , 0},
	 			{-5.92, 3.80, -0.6613, 33.1787 , 18.6666, 0.1683},
				{-0.24, -0.66,-0.6613, 51.8453, 7.2218, 0}
			}
	 	};
    
    % Marietta Track is divided into 5 regions of circular arcs and straight segments, CCRF is divided into 15 such segments
    if (map_flag_ == 1 || map_flag_ == 2)
        number_of_segments_ = 5;
    else
        number_of_segments_ = 15;
    end
%     track_length_ = map_parameters_{map_flag_}{number_of_segments_}{4} + map_parameters_{map_flag_}{number_of_segments_}{5};

    current_region_ = indentifyRegion(x, y, map_parameters_, map_flag_);
    prev_reg = current_region_ - 1;
    if (prev_reg == 0)
        prev_reg = number_of_segments_;
    end

    curvature_ = map_parameters_{map_flag_}{current_region_}{6};

    % Calculate the s, ey, epsi from x, y, yaw depending on whether the region is a straight segment or a circular arc
    if (curvature_ == 0)
        [epsi, ey, s] = calculateStraightSegment(map_parameters_, map_flag_, current_region_, prev_reg, x, y, yaw);
    else
        [epsi, ey, s] = calculateCurvedSegment(map_parameters_, map_flag_, current_region_, prev_reg, x, y, yaw);
    end
%     s_ = s;
%     
%     % Get the bias on s to ensure that when the controller starts, the s value is 0 indicating the start of the track
%     if (iterations_ == 0)
%         s_bias_ = s_;
% %         start_time_ = ros::Time::now().toSec();
% %         lap_start_time_ = ros::Time::now().toSec();
% %         lap_max_speed_ = 0;
% %         max_slip_ = 0;	
%     end
% 
%     % Get the maximum lap speed
% %     if (speed_ > lap_max_speed_)
% %         lap_max_speed_ = speed_;
% % 
% %     if (slip_ > max_slip_)
% %         max_slip_ = slip_;
%     path_s = s_ - s_bias_;
% 
%     if (path_s < 0)
%             path_s = path_s + track_length_;
%     end
end

function curvatures = get_curvature(s)
    map_flag_ = 2;
    map_parameters_ = ...
	 	{
	 		{
				{-8.62, 8.38, 2.36, 0, 5.9821, 0},					 
				{-16.94, -0.19, -0.80, 5.9821, 18.7621, 0.1674}, 
				{-8.81, -8.64, -0.80, 24.7552, 11.726, 0}, 
				{-0.12, 0.05, 2.36, 36.4702, 19.304, 0.1627}, 
				{-4.37, 4.17, 2.36, 55.774, 5.919, 0}
			},

			{
				{2.78,-2.97,-0.6613, 0, 3.8022, 0},
				{10.04,6.19, 2.4829, 3.8022, 18.3537,0.1712 }, 
				{1.46, 13.11,2.4829, 22.1559, 11.0228 , 0},
	 			{-5.92, 3.80, -0.6613, 33.1787 , 18.6666, 0.1683},
				{-0.24, -0.66,-0.6613, 51.8453, 7.2218, 0}
			}
	 	};
    all_params = cell2mat(vertcat(map_parameters_{2}{1:end}));
    starts = all_params(:,4);
    fin = all_params(end, 4) + all_params(end, 5);
    while any(s > fin)
        index = find(s > fin);
        s(index) = s(index) - fin;
    end
    s(end) = 4;
    gt = s >= starts;
    curvatures = zeros(1,size(gt, 2));
    for ii = 1:size(gt, 2)
        idx = find(gt(:,ii));
        current_region = idx(end)
        curvature = map_parameters_{map_flag_}{current_region}{6};
        curvatures(ii) = curvature;
    end
end

function region = indentifyRegion(x, y, map_parameters_, map_flag_)
    % Identify the region by dividing the map into a set of straight lines to denote different regions and finding where the point lies wrt these lines 
    if (map_flag_ == 2)
        if ( y <= (map_parameters_{map_flag_}{1}{2} - map_parameters_{map_flag_}{2}{2}) / (map_parameters_{map_flag_}{1}{1} - map_parameters_{map_flag_}{2}{1}) * (x - map_parameters_{map_flag_}{1}{1}) + map_parameters_{map_flag_}{1}{2})
            region = 1+1;
            return
        elseif ( y >= (map_parameters_{map_flag_}{4}{2} - map_parameters_{map_flag_}{3}{2}) / (map_parameters_{map_flag_}{4}{1} - map_parameters_{map_flag_}{3}{1}) * (x - map_parameters_{map_flag_}{3}{1}) + map_parameters_{map_flag_}{3}{2})
            region = 3+1;
            return
        else
            x01 = 0.5 * (map_parameters_{map_flag_}{1}{1} + map_parameters_{map_flag_}{2}{1});
            y01 = 0.5 * (map_parameters_{map_flag_}{1}{2} + map_parameters_{map_flag_}{2}{2});
            x23 = 0.5 * (map_parameters_{map_flag_}{3}{1} + map_parameters_{map_flag_}{4}{1});
            y23 = 0.5 * (map_parameters_{map_flag_}{3}{2} + map_parameters_{map_flag_}{4}{2});
            inc = (y01 - y23)/(x01 - x23);

            if (y >= inc * (x-x01) + y01)
                region = 2+1;
                return;
            elseif ( y <= - 1.0 /inc * (x - map_parameters_{map_flag_}{5}{1}) + map_parameters_{map_flag_}{5}{2})
                region = 0+1;
                return;
            else
                region = 4+1;
                return;
            end

        end
    end
end

function [d_psi_, n_, s_] = calculateStraightSegment(map_parameters_, map_flag_, curr_reg, prev_reg, x, y, psi)
    d_rel = sqrt( power(x - map_parameters_{map_flag_}{prev_reg}{1}, 2)  + power(y - map_parameters_{map_flag_}{prev_reg}{2}, 2) );
    theta_rel = atan2(y - map_parameters_{map_flag_}{prev_reg}{2}, x - map_parameters_{map_flag_}{prev_reg}{1} ) - ...
						   atan2(map_parameters_{map_flag_}{curr_reg}{2} - map_parameters_{map_flag_}{prev_reg}{2}, map_parameters_{map_flag_}{curr_reg}{1} - map_parameters_{map_flag_}{prev_reg}{1});

    if (theta_rel < -pi)
        theta_rel = theta_rel + 2.0 * pi;
    elseif (theta_rel > pi)
        theta_rel = theta_rel - 2.0 * pi;
    end

    s_ = map_parameters_{map_flag_}{curr_reg}{4} + d_rel * cos(theta_rel);
    n_ = d_rel * sin(theta_rel);
    d_psi_ = psi - map_parameters_{map_flag_}{curr_reg}{3};

    if (d_psi_ < -pi)
        while (d_psi_ < -pi)
            d_psi_ = d_psi_ + 2.0 * pi;
        end
    elseif (d_psi_ > pi)
        while (d_psi_ > pi)
            d_psi_ = d_psi_ - 2.0 * pi;
        end
    end
end

function [d_psi_, n_, s_] = calculateCurvedSegment(map_parameters_, map_flag_, curr_reg, prev_reg, x, y, psi)
    x_c = 0.5 * (map_parameters_{map_flag_}{curr_reg}{1} + map_parameters_{map_flag_}{prev_reg}{1});
	y_c = 0.5 * (map_parameters_{map_flag_}{curr_reg}{2} + map_parameters_{map_flag_}{prev_reg}{2});
		
    d_rel = sqrt( power(x - x_c, 2) + power(y - y_c, 2) );
    theta_rel = atan2(y - y_c, x - x_c) - atan2(map_parameters_{map_flag_}{prev_reg}{2} - y_c, map_parameters_{map_flag_}{prev_reg}{1} - x_c);

    if (theta_rel < -pi)
        theta_rel = theta_rel + 2.0 * pi;
    elseif (theta_rel > pi)
        theta_rel = theta_rel - 2.0 * pi;
    end

    s_ = map_parameters_{map_flag_}{curr_reg}{4} + theta_rel / map_parameters_{map_flag_}{curr_reg}{6};
    n_ = 1.00 / map_parameters_{map_flag_}{curr_reg}{6} - d_rel;
    d_psi_ = psi - map_parameters_{map_flag_}{prev_reg}{3} - theta_rel;

    if (d_psi_ < -pi)
        while (d_psi_ < -pi)
            d_psi_ = d_psi_ + 2.0 * pi;
        end
    elseif (d_psi_ > pi)
        while (d_psi_ > pi)
            d_psi_ = d_psi_ - 2.0 * pi;
        end
    end
end