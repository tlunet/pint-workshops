%% Settings for Matlab
close all; clear;
set( 0, 'defaultAxesTickLabelInterpreter', 'latex' );
set( 0, 'defaultLegendInterpreter',        'latex' );
set( 0, 'defaultTextInterpreter',          'latex' );
set( 0, 'defaultAxesFontSize', 12 );
format long 

%% Numerical settings
T = 20;
nStep = 20000;

% Parameters and position of fixed points
sigma = 10; rho=28; beta = 8/3;
K = (beta*(rho-1))^0.5;
xf = K; yf = K; zf = rho-1;

% Definition of the function used for time integration
f=@(t,u) LorenzOperator(u(1), u(2), u(3), sigma, rho, beta);

% Baseling solve
[t1, u1] = ForwardEuler(f, 0, T, [20, 5, -5], nStep);

% Small variation on the initial solution
epsilon1 = 1e-3;
[t2, u2] = ForwardEuler(f, 0, T, [20+epsilon1, 5, -5], nStep);

% Smaller variation on the initial solution
epsilon2 = 1e-10;
[t3, u3] = ForwardEuler(f, 0, T, [20+epsilon2, 5, -5], nStep);

%% Q-1(f): Displaying animation
plot3(xf,yf,zf,'ok');
hold on;
plot3(-xf,-yf,zf,'ok');

grid on
axis([-20 30 -30 40 -10 60]); view([-13,8]);
xlabel('$x$'); ylabel('$y$'); zlabel('$z$'); 

curve1 = animatedline('Color','blue');
for i=1:5:length(t1)
    p1 = plot3(u1(1,i), u1(2,i), u1(3,i),'sb', 'MarkerSize', 6);
    addpoints(curve1, u1(1,i), u1(2,i), u1(3,i));
    drawnow
    pause(0.001);
    delete(p1);
end
hold off;

%% Q-1(g): Displaying animation of the two trajectories
plot3(xf,yf,zf,'ok');
hold on;
plot3(-xf,-yf,zf,'ok');

grid on
axis([-20 30 -30 40 -10 60]); view([-13,8]);
xlabel('$x$'); ylabel('$y$'); zlabel('$z$');

curve1 = animatedline('Color','blue');
curve2 = animatedline('Color','red');
for i=1:5:length(t1)
    p1 = plot3(u1(1,i), u1(2,i), u1(3,i),'sb', 'MarkerSize', 12);
    p2 = plot3(u2(1,i), u2(2,i), u2(3,i),'or');
    addpoints(curve1, u1(1,i), u1(2,i), u1(3,i));
    addpoints(curve2, u2(1,i), u2(2,i), u2(3,i));
    drawnow
    pause(0.001);
    delete(p1); delete(p2);
end
hold off;

%% Q-1(g): Displaying the two x-trajectories
plot(t1, u1(1,:), 'DisplayName', 'Original');
hold on;
plot(t2, u2(1,:), 'DisplayName', sprintf('%1.0g variation', epsilon1));
plot(t3, u3(1,:), 'DisplayName', sprintf('%1.0g variation', epsilon2));
xlabel('Time'); ylabel('$x(t)$'); legend();
hold off;

%% Q-1(g): Displaying the difference between x-trajectories
semilogy(t1, abs(u1(1,:)-u2(1,:)), ...
    'DisplayName', sprintf('%1.0g variation', epsilon1));
hold on;
semilogy(t1, abs(u1(1,:)-u3(1,:)), ...
    'DisplayName', sprintf('%1.0g variation', epsilon2));
xlabel('Time'); ylabel('Difference'); legend();
hold off; grid on;

