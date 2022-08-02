T = 1;
y0 = 1;
nStep = 10;

lC = get(gca,'colororder');
close;

labOption = 'DisplayName';

labelTh = 'Theorique';
labelFE = 'Forward Euler';
labelBE = 'Backward Euler';

vLam = [1j, 1j-1];
vStyle = {'-', '--'};

for i=1:2
    lam = vLam(i); s = vStyle{i};
    [~, yFE] = DahlquistFE(lam, y0, T, nStep);
    [t, yBE] = DahlquistBE(lam, y0, T, nStep);
    yTh = exp(lam*t);

    figure(1); hold on;
    plot(real(yTh), imag(yTh), [s,'o'],...
        'Color', lC(1,:), labOption, labelTh);
    plot(real(yFE), imag(yFE), [s,'s'],...
        'Color', lC(2,:), labOption, labelFE);
    plot(real(yBE), imag(yBE), [s,'^'],...
        'Color', lC(3,:), labOption, labelBE);

    figure(2); hold on;
    plot(t, abs(yTh), [s,'o'],...
        'Color', lC(1,:), labOption, labelTh);
    plot(t, abs(yFE), [s,'s'],...
        'Color', lC(2,:), labOption, labelFE);
    plot(t, abs(yBE), [s,'^'],...
        'Color', lC(3,:), labOption, labelBE);

    figure(3); hold on;
    plot(t, angle(yTh), [s,'o'],...
        'Color', lC(1,:), labOption, labelTh);
    plot(t, angle(yFE), [s,'s'],...
        'Color', lC(2,:), labOption, labelFE);
    plot(t, angle(yBE), [s,'^'],...
        'Color', lC(3,:), labOption, labelBE);
    
    labOption = 'HandleVisibility';
    labelTh = 'off'; labelFE = 'off'; labelBE = 'off';
end

figName = {'traj', 'vabs', 'angle'};
for i=1:3
    fig = figure(i);
    set(fig, 'NumberTitle', 'off', 'Name', figName{i});
    grid on;
    legend({});
end

figure(1)
xlim([-0.1, 1.1])
xlabel('Re(y)')
ylabel('Im(y)')
axis equal;

for i=2:3
    figure(i);
    xlabel('Time')
end