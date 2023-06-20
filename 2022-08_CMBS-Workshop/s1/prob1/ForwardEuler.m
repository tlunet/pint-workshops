function [t,u]=ForwardEuler(f,t0,tEnd,u0,nStep)
% FORWARDEULER solves the ODE du/dt=f(t,u) using forward Euler
%  
%   Parameters
%   ----------
%   f : function
%       The :math:`f` operator, as a function that take a scalar **t** and
%       a vector **u** as argument,
%       and returns a vector of the same size as **u**.
%   t0 : float
%       The time of the initial solution
%   tEnd : float
%       The time of the final solution
%   u0 : vector
%       The initial solution
%   nStep : int
%       The number of time step to be performed
% 
%  Returns
%  -------
%  t : vector of size :math:`N_{step}+1`
%      The discrete times from **t0** to **tEnd** when the solution was
%      computed (including the initial time).
%  u : matrix of size :math:`(N_{dof} \\times N_{step}+1)`
%      The solution of the ODE at each time steps, including the initial time.

nDOF = length(u0);
u = zeros(nDOF, nStep+1);
t = linspace(t0, tEnd, nStep+1);

u(:, 1) = u0;
dt = (tEnd-t0)/nStep;

for i=1:nStep               % Update the solution at each time-steps
    u(:, i+1) = u(:, i) + dt*f(t(i), u(:, i));
end

