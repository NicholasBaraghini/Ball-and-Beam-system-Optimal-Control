%% Ball & Beam Parameters
% come riferimento è stata presa una pallina di acciaio nichelato stile
% geomag 

density= 7.5; %kg/dm^3

r = 0.0127; % in dm

m = density*4/3*pi*r^3;

g = 9.81; 

% J = 0.1125;

Jb = 2/5*m*r/10;

d1 = 1/(Jb/(r^2)+m);

M = 0.1; % in kg

L = 1; % in m

J= M/12;

K = 100;

T1 = 1;

