% Plot stability region of Forward Euler
theta = linspace(0, 2*pi, 10000);
circle = -1+exp(1i*theta);
plot(real(circle), imag(circle), 'HandleVisibility','off');

% Lambda with their associated dtMax
vLam = [-1, 1j, 1j-2];
vDtMax = [2, 1.1, 4./5];

% Plot each curve
hold on
for i=1:length(vLam)
    dt = linspace(0, vDtMax(i)) ;
    r = dt*vLam(i) ;
    plot(real(r), imag(r), 'DisplayName', ['$\lambda=$', num2str(vLam(i))]);
end

% Axis general settings
legend({}, 'Interpreter','latex')
axis equal
grid on