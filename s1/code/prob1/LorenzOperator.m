function fxyz = LorenzOperator(x, y, z, sigma, rho, beta)
% LORENZ Evaluate the Lorenz operator on a given position vector
%   Parameters
%   ----------
%   x : float
%       The :math:`x` position
%   y : float
%       The :math:`y` position
%   z : float
%       The :math:`z` position
%   sigma : float
%       The :math:`\\sigma` parameter
%   rho : float
%       The :math:`\\rho` parameter
%   beta : float
%       The :math:`\\beta` parameter
% 
%   Returns
%   -------
%   fxyz : vector of size 3
%       The result
fxyz = [sigma*(y-x); x*(rho-z)-y; x*y-beta*z];
end

