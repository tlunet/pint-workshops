function [t,y] = DahlquistBE(lam, y0, T, nStep)
%DAHLQUISTBE  Solve the Dahlquist equation with Backward Euler
%   TODO
dt = T/nStep;
g = 1/(1-lam*dt);
p = 0:nStep;
y = g.^p*y0;
t = linspace(0, T, nStep+1);