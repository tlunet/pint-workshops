% Settings for Matlab
close all; clear;
set( 0, 'defaultAxesTickLabelInterpreter', 'latex' );
set( 0, 'defaultLegendInterpreter',        'latex' );
set( 0, 'defaultTextInterpreter',          'latex' );
set( 0, 'defaultAxesFontSize', 12 );
format long 

% Definition of the function used for time integration
sigma = 10; rho=28; beta = 8/3;
f=@(t,u) LorenzOperator(u(1), u(2), u(3), sigma, rho, beta);

% Local error computation
nStep = 1;
nStepRef = 1000;
dt = [0.0001, 0.001, 0.01, 0.1];
lErrLoc = [];
for T = dt
    [~, u] = ForwardEuler(f, 0, T, [20, 5, -5], nStep);
    [~, uRef] = ForwardEuler(f, 0, T, [20, 5, -5], nStepRef);
    err = norm(uRef(:, end) - u(:, end));
    lErrLoc = [lErrLoc, err];
end

% Global error computation
T = max(dt);
lStep = T./dt;
nStepRef = 100*max(lStep);
[t, uRef] = ForwardEuler(f, 0, T, [20, 5, -5], nStepRef);
lErrGlob = [];
for nStep = lStep
    [~, u] = ForwardEuler(f, 0, T, [20, 5, -5], nStep);
    err = norm(uRef(:, end) - u(:, end));
    lErrGlob = [lErrGlob, err];
end

% Q-1(h): error plot
loglog(dt, lErrLoc, '--o', 'DisplayName', 'Loc. trun. error');
hold on;
loglog(dt, lErrGlob, '--s', 'DisplayName', 'Glob. trun. error');
loglog(dt, 10^3.5 * dt, 'DisplayName', 'Order 1');
loglog(dt, 10^3 * dt.^2, 'DisplayName', 'Order 1');
xlabel('$\Delta t$'); legend('Location', 'northwest'); grid on;
