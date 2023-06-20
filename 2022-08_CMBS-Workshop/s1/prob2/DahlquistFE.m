function [t,y] = DahlquistFE(lam, y0, T, nStep)
%DAHLQUISTBE Solve the Dahlquist equation with Forward Euler
%   TODO
dt = T/nStep;
g = 1+lam*dt;
p = 0:nStep;
y = g.^p*y0;
t = linspace(0, T, nStep+1);

