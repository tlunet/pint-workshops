% Settings for Matlab
close all; clear;
set( 0, 'defaultAxesTickLabelInterpreter', 'latex' );
set( 0, 'defaultLegendInterpreter',        'latex' );
set( 0, 'defaultTextInterpreter',          'latex' );
set( 0, 'defaultAxesFontSize', 12 );
format long 

% Parameters
nStep = 64;
T = 1/2;
method = 'BE';

% Parameters for numerical solution
L = 1;
J = 99;
A = FinDiffMatrixD2C2(J, L);
x = linspace(0, 1, J+2);
x = x(2:end-1);

% Parameters for analytical solution
nF = 5;
n = 0:nF-1;
global nMesh xMesh;
[xMesh, nMesh] = meshgrid(x, n);

% Definition of the source term
b=@(t) 0;

u0 = 20*ones(J, 1);

if strcmp(method, 'FE')
    [t, u] = ForwardEulerLin(A, u0, b, 0, T, nStep);
elseif strcmp(method, 'BE')
    [t, u] = BackwardEulerLin(A, u0, b, 0, T, nStep);
end

plot(x, u(:, int64(nStep/4)+1), 'DisplayName', '$T=1/8$');
hold on;
plot(x, u(:, int64(nStep/2)+1), 'DisplayName', '$T=1/4$');
plot(x, u(:, nStep+1), 'DisplayName', '$T=1/2$');
gray = [0.5, 0.5, 0.5];
plot(x, uTh(1/8), 'o', 'Color', gray, 'HandleVisibility', 'off', ...
    'MarkerIndices',1:5:length(x));
plot(x, uTh(1/4), 'o', 'Color', gray, 'HandleVisibility', 'off', ...
    'MarkerIndices',1:5:length(x))
plot(x, uTh(1/2), 'o', 'Color', gray, 'HandleVisibility', 'off', ...
    'MarkerIndices',1:5:length(x))
xlabel('$x$'); legend(); grid on;

% Function to compute the analytical solution at time t
function sol=uTh(t)
    global nMesh xMesh; 
    s = (80/pi)./(2*nMesh+1) .* exp(-(2*nMesh+1).^2 * pi^2 * t) ...
        .* sin((2*nMesh+1).*pi .*xMesh);
    sol = sum(s, 1);
end