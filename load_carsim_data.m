%% interpolate and convert units
dt = 0.001;
t = Time;
time = t(1):dt:t(end);

mthd='pchip';
vx=interp1(t,Vx,time,mthd);
vy=interp1(t,Vy,time,mthd);
wz=interp1(t,AVz,time,mthd);
wF=interp1(t,(AVy_L1+AVy_R1)/2,time,mthd);
wR=interp1(t,(AVy_L2+AVy_R2)/2,time,mthd);
wLF=interp1(t,AVy_L1,time,mthd);
wLR=interp1(t,AVy_L2,time,mthd);
wRF=interp1(t,AVy_R1,time,mthd);
wRR=interp1(t,AVy_R2,time,mthd);
yaw=interp1(t,Yaw,time,mthd);
X=interp1(t,Xcg_SM,time,mthd);
Y=interp1(t,Ycg_SM,time,mthd);
% steering = min(abs(Steer_L1), abs(Steer_R1)) .* sign(StrSWr_1);
% steering=interp1(t,Steer_SW,time,mthd);
% steeringL=interp1(t,(Steer_L1),time,mthd);
% steeringR=interp1(t,Steer_R1,time,mthd);
% ffx=interp1(t,Fx_A1,time,mthd);
% frx=interp1(t,Fx_A2,time,mthd);
% frfx=interp1(t,Fx_R1,time,mthd);
% frrx=interp1(t,Fx_R2,time,mthd);
% ffy=interp1(t,Fy_A1,time,mthd);
% fry=interp1(t,Fy_A2,time,mthd);
% frfy=interp1(t,Fy_R1,time,mthd);
% frry=interp1(t,Fy_R2,time,mthd);
% beta=interp1(t,Beta,time,mthd);
% alphalf=interp1(t,Alpha_L1,t,mthd);
% alphalr=interp1(t,Alpha_L2,t,mthd);
% alpharf=interp1(t,Alpha_R1,t,mthd);
% alpharr=interp1(t,Alpha_R2,t,mthd);
% FzF=interp1(t,Fz_A1,time,mthd);
% FzR=interp1(t,Fz_A2,time,mthd);
% muFx=interp1(t,(MuX_L1+MuX_R1),time,mthd);
% muRx=interp1(t,(MuX_L2+MuX_R2),time,mthd);
% muFy=interp1(t,(MuY_L1+MuY_R1),time,mthd);
% muRy=interp1(t,(MuY_L2+MuY_R2),time,mthd);

vx = vx/60/60*1000;
vy = vy/60/60*1000;
wz = wz/180*pi;
wF = wF/60*2*pi;
wR = wR/60*2*pi;
wLF = wLF/60*2*pi;
wLR = wLR/60*2*pi;
wRF = wRF/60*2*pi;
wRR = wRR/60*2*pi;
yaw = yaw/180*pi;
% steering = steering/180*pi;
% steeringL = steeringL/180*pi;
% steeringR = steeringR/180*pi;
% beta = beta/180*pi;
% alphalf=alphalf/180*pi;
% alphalr=alphalr/180*pi;
% alpharf=alpharf/180*pi;
% alpharr=alpharr/180*pi;

carsim_states = [vx; vy; wz; wF; wR; yaw; X; Y];% wLF; wLR; wRF; wRR];
% states = [vx; vy; wz; yaw];
% inputs = [steering; steeringL; steeringR; FzF; FzR; ffx; ffy; frx; fry];

%%
N = 2000;
Nf = 2000;
% subset_states = states(:,N:176000);
% subset_inputs = inputs(:,N:10:end-Nf);
% subset_ff = ff(:,N/10:end-Nf/10);
state = carsim_states(:,1);
% carsim_states(7:8,N) = states(7:8,N);
% time = (1:length(subset_states))/100;%1:20000;
% analytic_states = zeros(length(state), length(time));
% forces = zeros(8, length(time));
% ff_training_data = zeros(5, length(time));

figure;
for ii = 1:length(state)
    subplot(length(state)/2, 2, ii);
    hold on
%     plot(time(1:end-Nf), subset_states(ii, 1:end-Nf))
%     if ii < 7
%     else
%         plot(time, analytic_states(ii, :))
%     end
    plot(time(1:end-Nf), carsim_states(ii, 1:end-Nf))

end
%%
tire_B = 10;
tire_C = 2.0;
tire_D = 1.18;
a = 0.75;
figure;
subplot(4,2,1);
sx = -Kappa_L1./(1+Kappa_L1);
sy = tan(alphalf)./(1+Kappa_L1);
s = sqrt(sx.^2 + sy.^2);
plot(sx, Fx_L1./Fz_L1, '.');
hold on;
mu = tire_D * sin(tire_C * atan(tire_B * s));
mux = -sx ./ s .* mu;
plot(sx, mux, '.');
subplot(4,2,2);
sx = -Kappa_R1./(1+Kappa_R1);
sy = tan(alpharf)./(1+Kappa_R1);
s = sqrt(sx.^2 + sy.^2);
plot(sx, Fx_R1./Fz_R1, '.');
hold on;
mu = tire_D * sin(tire_C * atan(tire_B * s));
mux = -sx ./ s .* mu;
plot(sx, mux, '.');
subplot(4,2,3);
sx = -Kappa_L2./(1+Kappa_L2);
sy = tan(alphalr)./(1+Kappa_L2);
s = sqrt(sx.^2 + sy.^2);
plot(sx, Fx_L2./Fz_L2, '.');
hold on;
mu = tire_D * sin(tire_C * atan(tire_B * s));
mux = -sx ./ s .* mu;
plot(sx, mux, '.');
subplot(4,2,4);
sx = -Kappa_R2./(1+Kappa_R2);
sy = tan(alpharr)./(1+Kappa_R2);
s = sqrt(sx.^2 + sy.^2);
plot(sx, Fx_R2./Fz_R2, '.');
hold on;
mu = tire_D * sin(tire_C * atan(tire_B * s));
mux = -sx ./ s .* mu;
plot(sx, mux, '.');
% y
subplot(4,2,5);
sx = -Kappa_L1./(1+Kappa_L1);
sy = tan(alphalf)./(1+Kappa_L1);
s = sqrt(sx.^2 + sy.^2);
plot(sy, Fy_L1./Fz_L1, '.');
hold on;
mu = tire_D * sin(tire_C * atan(tire_B * s));
muy = -a*sy ./ s .* mu;
plot(sy, muy, '.');
subplot(4,2,6);
sx = -Kappa_R1./(1+Kappa_R1);
sy = tan(alpharf)./(1+Kappa_R1);
s = sqrt(sx.^2 + sy.^2);
plot(sy, Fy_R1./Fz_R1, '.');
hold on;
mu = tire_D * sin(tire_C * atan(tire_B * s));
muy = -a*sy ./ s .* mu;
plot(sy, muy, '.');
subplot(4,2,7);
sx = -Kappa_L2./(1+Kappa_L2);
sy = tan(alphalr)./(1+Kappa_L2);
s = sqrt(sx.^2 + sy.^2);
plot(sy, Fy_L2./Fz_L2, '.');
hold on;
mu = tire_D * sin(tire_C * atan(tire_B * s));
muy = -a*sy ./ s .* mu;
plot(sy, muy, '.');
subplot(4,2,8);
sx = -Kappa_R2./(1+Kappa_R2);
sy = tan(alpharr)./(1+Kappa_R2);
s = sqrt(sx.^2 + sy.^2);
plot(sy, Fy_R2./Fz_R2, '.');
hold on;
mu = tire_D * sin(tire_C * atan(tire_B * s));
muy = -a*sy ./ s .* mu;
plot(sy, muy, '.');