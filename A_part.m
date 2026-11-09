%% Three-node DC microgrid: equilibrium and spectral-radius analysis

clearvars; clc;

% outputDir = fullfile(fileparts(mfilename('fullpath')), 'output');
% if ~isfolder(outputDir), mkdir(outputDir); end

%% Three-node system parameters
cfg.three.u0 = 1;
cfg.three.g0L = 1;
cfg.three.g0G = 0.15;
cfg.three.gLG = 1.2;
cfg.three.PG = -0.3;
cfg.three.scanNumber = 201;
cfg.three.figureLoads = [0.45, 0.90, 1, 1.10];

p3 = cfg.three;

%% Three-node model and equilibrium calculation
edges3 = [1, 2, p3.g0L; 1, 3, p3.g0G; 2, 3, p3.gLG];
model3 = build_dc_model(3, edges3, 1, p3.u0, zeros(3,1));

[PLcritical, uCritical] = three_critical_load(p3);
alpha3 = unique(sort([linspace(0.001, 1.12, p3.scanNumber), p3.figureLoads]));
PLscan = PLcritical * alpha3;

records = zeros(0, 3);  % [PL, spectral radius, high-voltage branch]

for k = 1:numel(PLscan)
    PL = PLscan(k);
    U = three_all_positive_roots(p3, PL);

    for j = 1:size(U, 2)
        x = U(:,j).^2;
        J = icm_jacobian(x, model3);
        rho = sparse_perron(J, x, model3.d);
        isHigh = (j == size(U, 2));
        records(end+1,:) = [PL, rho, isHigh];
    end
end

rhoCritical3 = sparse_perron(icm_jacobian(uCritical.^2, model3), ...
    uCritical.^2, model3.d);

equilibria3 = array2table(records, ...
    'VariableNames', {'PL', 'rho', 'isHigh'});

%% Generate and export figures
draw_figure1(p3, PLcritical, uCritical);
draw_figure2(equilibria3, PLcritical, rhoCritical3);

% exportgraphics(figure(1), fullfile(outputDir, 'Figure1.png'), 'Resolution', 180);
% exportgraphics(figure(2), fullfile(outputDir, 'Figure2.png'), 'Resolution', 180);

%% Local functions
function [R1, R2] = three_residual(vL, vG, s, PL)
% Original nodal power-balance equations.
% Positive power denotes consumption; negative power denotes generation.

R1 = (s.g0L+s.gLG)*vL.^2 - s.gLG*vL.*vG ...
    - s.g0L*s.u0*vL + PL;
R2 = (s.g0G+s.gLG)*vG.^2 - s.gLG*vL.*vG ...
    - s.g0G*s.u0*vG + s.PG;
end

function m = build_dc_model(n, edges, S, uS, Gbus)
% Construct the conductance Laplacian and partition the nodal matrix.

ne = size(edges,1);
S = S(:);
L = setdiff((1:n)', S, 'stable');

B = sparse([(1:ne)'; (1:ne)'], [edges(:,1); edges(:,2)], ...
    [ones(ne,1); -ones(ne,1)], ne, n);
Ynetwork = B' * spdiags(edges(:,3), 0, ne, ne) * B;
Ybus = Ynetwork + spdiags(Gbus, 0, n, n);

m.n = numel(L);
m.uS = uS(:);
YLL = Ybus(L,L);
YLS = Ybus(L,S);
m.d = diag(YLL);
Y1 = YLL - spdiags(m.d, 0, m.n, m.n);
m.A = -spdiags(1./m.d, 0, m.n, m.n) * Y1;
m.b = -(YLS*m.uS)./m.d;
end

function J = icm_jacobian(x, m)
% Jacobian of the ICM mapping.

u = sqrt(x);
D = spdiags(u, 0, m.n, m.n);
W = D * m.A * D;
J = 0.5 * (W + spdiags(sum(W,2) + m.b.*u, 0, m.n, m.n)) ...
    * spdiags(1./x, 0, m.n, m.n);
end

function rho = sparse_perron(J, x, d)
% Compute the Perron eigenvalue using a symmetric similarity transform.

nn = size(J,1);
scale = sqrt(d./x);
H = spdiags(scale, 0, nn, nn) * J * spdiags(1./scale, 0, nn, nn);
H = (H + H')/2;

opts.tol = 1e-12;
opts.maxit = 3000;
opts.v0 = ones(nn,1)/sqrt(nn);
[~, lambda] = eigs(H, 1, 'largestreal', opts);
rho = real(lambda(1,1));
end

function [p, base] = three_polynomial(s, PL)
% Eliminate vL and obtain the polynomial equation in vG.

a = s.g0L + s.gLG;
d = s.g0G + s.gLG;
c = s.gLG;
b = s.g0L*s.u0;
t = s.g0G*s.u0;
N = [d, -t, s.PG];

base = c^2*[N, 0, 0] + b*c*[0, N, 0] - a*conv(N, N);
p = base - PL*c^2*[0, 0, 1, 0, 0];
end

function [PLcrit, uc] = three_critical_load(s)
% Calculate the critical load from the stationary point of PL(vG).

[~, base] = three_polynomial(s, 0);
d = s.g0G + s.gLG;
t = s.g0G*s.u0;
c = s.gLG;

wmin = (t + sqrt(t^2 - 4*d*s.PG))/(2*d);
rr = roots(base.*[2, 1, 0, -1, -2]);
ws = real(rr(abs(imag(rr)) < 1e-9 & real(rr) > wmin));

w = ws(1);
PLcrit = polyval(base, w)/(c^2*w^2);
uc = [(d*w^2 - t*w + s.PG)/(c*w); w];
end

function U = three_all_positive_roots(s, PL)
% Enumerate all positive real roots of the reduced polynomial.

p = three_polynomial(s, PL);
p = p/max(abs(p));
rr = roots(polyder(p));
cr = real(rr(abs(imag(rr)) < 1e-9));

d = s.g0G + s.gLG;
t = s.g0G*s.u0;
c = s.gLG;
wmin = (t + sqrt(t^2 - 4*d*s.PG))/(2*d);
bound = max(wmin+1, 1+max(abs(p(2:end)/p(1))));
cuts = sort([wmin; cr(cr > wmin & cr < bound); bound]);
candidates = [];

for k = 1:numel(cuts)
    w = cuts(k);
    if abs(polyval(p,w)) <= 3e-13*max(1,polyval(abs(p),w))
        candidates(end+1) = w;
    end
end

for k = 1:numel(cuts)-1
    l = cuts(k);
    r = cuts(k+1);
    fl = polyval(p,l);
    fr = polyval(p,r);
    nearL = abs(fl) <= 3e-13*max(1,polyval(abs(p),l));
    nearR = abs(fr) <= 3e-13*max(1,polyval(abs(p),r));

    if ~nearL && ~nearR && sign(fl) ~= sign(fr)
        candidates(end+1) = fzero(@(w) polyval(p,w), [l,r], ...
            optimset('TolX',1e-13,'Display','off'));
    end
end

U = zeros(2,0);
for w = sort(candidates)
    v = (d*w^2 - t*w + s.PG)/(c*w);
    if v <= 1e-9 || w <= 0
        continue;
    end
    if isempty(U) || all(max(abs(U-[v;w]),[],1) > 1e-7*max(1,w))
        U(:,end+1) = [v;w];
    end
end
end

function draw_figure1(s, PLc, uc)
% Plot the equilibrium curves and the regions associated with the CPL load.

f = figure(1);
clf(f);
set(f, 'Name', 'Figure 1: CPL load increase in one plot', ...
    'Color', 'w', 'Position', [40,50,1080,880]);
ax = axes('Parent', f);
hold(ax, 'on');

ratios = s.figureLoads;
Ulow = three_all_positive_roots(s, ratios(1)*PLc);
vmax = 1.18*max(Ulow(:));

PLs = ratios*PLc;
allU = cell(1,4);
rootsForGrid = uc;

for k = 1:4
    U = three_all_positive_roots(s, PLs(k));
    allU{k} = U;
    rootsForGrid = [rootsForGrid, U]; 
end

vs = linspace(0.0001, vmax, 1000);
[VS, GS] = meshgrid(vs, vs);
[Rlow, Rfixed] = three_residual(VS, GS, s, PLs(1));
Rtop = three_residual(VS, GS, s, PLs(4));
band = (Rlow <= 0 & Rtop >= 0);
shadeTol = 1e-11*max([1, abs(PLs(1)), abs(s.PG)]);
strict = (Rlow < -shadeTol & Rfixed < -shadeTol);

blue = [0.28,0.53,0.90];
green = [0.30,0.72,0.38];

image(ax, [vs(1),vs(end)], [vs(1),vs(end)], ...
    repmat(reshape(blue,1,1,3),numel(vs),numel(vs)), ...
    'AlphaData', 0.18*double(band), 'HandleVisibility', 'off');
image(ax, [vs(1),vs(end)], [vs(1),vs(end)], ...
    repmat(reshape(green,1,1,3),numel(vs),numel(vs)), ...
    'AlphaData', 0.26*double(strict), 'HandleVisibility', 'off');
set(ax, 'YDir', 'normal');

hBand = patch(ax, NaN, NaN, blue, 'FaceAlpha', 0.18, 'EdgeColor', 'none');
hStrict = patch(ax, NaN, NaN, green, 'FaceAlpha', 0.26, 'EdgeColor', 'none');

vx = unique(sort([linspace(0.0001,vmax,1000), rootsForGrid(1,:)]));
vy = unique(sort([linspace(0.0001,vmax,1000), rootsForGrid(2,:)]));
[VL, VG] = meshgrid(vx, vy);

lineStyles = {'-','--','--','-'};
colors = [0.08,0.35,0.76; 0.35,0.55,0.82; ...
          0.24,0.40,0.65; 0.03,0.20,0.47];
hCPL = gobjects(1,4);

for k = 1:4
    R1 = three_residual(VL, VG, s, PLs(k));
    [~, hCPL(k)] = contour(ax, VL, VG, R1, [0,0], ...
        'Color', colors(k,:), 'LineWidth', 2, 'LineStyle', lineStyles{k});

    U = allU{k};
    if ~isempty(U)
        if k == 2
            plot(ax, U(1,:), U(2,:), 'o', 'Color', colors(k,:), ...
                'MarkerFaceColor', 'w', 'MarkerSize', 7, ...
                'LineWidth', 1.5, 'HandleVisibility', 'off');
        elseif k == 3
            plot(ax, uc(1), uc(2), 'kd', ...
                'MarkerFaceColor', [0.95,0.65,0.12], ...
                'MarkerSize', 9, 'HandleVisibility', 'off');
        else
            plot(ax, U(1,:), U(2,:), 'ko', ...
                'MarkerFaceColor', 'k', 'MarkerSize', 7, ...
                'HandleVisibility', 'off');
        end
    end
end

[~, R2] = three_residual(VL, VG, s, PLs(1));
[~, hMPPT] = contour(ax, VL, VG, R2, [0,0], ...
    'Color', [0.82,0.17,0.11], 'LineWidth', 2.3);

text(ax, Ulow(1,1)+0.025*vmax, Ulow(2,1)-0.045*vmax, ...
    'low voltage solution', 'FontSize', 10, ...
    'BackgroundColor', 'w', 'Margin', 1);
text(ax, Ulow(1,2)+0.025*vmax, Ulow(2,2)-0.055*vmax, ...
    'high voltage solution', 'HorizontalAlignment', 'left', ...
    'FontSize', 10, 'BackgroundColor', 'w', 'Margin', 1);
text(ax, uc(1)+0.015*vmax, uc(2)+0.12*vmax, 'critical point', ...
    'HorizontalAlignment', 'right', 'FontSize', 11, ...
    'BackgroundColor', 'w', 'Margin', 1);

va = 0.30*vmax;
a = s.g0L+s.gLG;
b = s.g0L*s.u0;
ya = (a*va-b+PLs(1)/va)/s.gLG;
yb = (a*va-b+PLs(4)/va)/s.gLG;
quiver(ax, va, ya+0.08*(yb-ya), 0, 0.80*(yb-ya), 0, ...
    'Color', [0.20,0.27,0.40], 'LineWidth', 1.5, ...
    'MaxHeadSize', 0.25, 'HandleVisibility', 'off');
text(ax, va+0.025*vmax, ya+0.25*(yb-ya), 'P_L increases', ...
    'Rotation', 90, 'FontSize', 10, 'Color', [0.20,0.27,0.40], ...
    'HorizontalAlignment', 'center');

labels = {sprintf('R_1: %.2f P_{L,crit} (initial)',ratios(1)), ...
    sprintf('R_1: %.2f P_{L,crit} (intermediate)',ratios(2)), ...
    'R_1: P_{L,crit} (critical)', ...
    sprintf('R_1: %.2f P_{L,crit} (final, no roots)',ratios(4)), ...
    'R_2: fixed P_G (MPPT)', ...
    'Region swept by R_1', ...
    'F(x)>x at the initial load'};

legend(ax, [hCPL,hMPPT,hBand,hStrict], labels, ...
    'Location', 'southeast', 'FontSize', 9);
xlim(ax, [0,vmax]);
ylim(ax, [0,vmax]);
axis(ax, 'square');
grid(ax, 'on');
box(ax, 'on');
set(ax, 'FontSize', 11, 'Layer', 'top');
xlabel(ax, 'v_L (p.u.)');
ylabel(ax, 'v_G (p.u.)');
title(ax, {'Figure 1: increasing CPL load - two equilibria, merging, then no equilibrium', ...
    sprintf('g_{0L}=%.2g, g_{0G}=%.2g, g_{LG}=%.2g, fixed P_G=%.2g; P_{L,crit}=%.8f', ...
    s.g0L,s.g0G,s.gLG,s.PG,PLc)});
end

function draw_figure2(T, PLc, rhoC)
% Plot the spectral radii of the high- and low-voltage equilibrium branches.

f = figure(2);
clf(f);
set(f, 'Name', 'Figure 2: Three-node spectral radius', ...
    'Color', 'w', 'Position', [80,80,1280,550]);
tiledlayout(1,2, 'TileSpacing', 'compact', 'Padding', 'compact');

hi = T.isHigh == 1 & T.PL < PLc-1e-10;
lo = T.isHigh == 0;

for j = 1:2
    nexttile;
    hold on;

    plot([T.PL(hi);PLc], [T.rho(hi);rhoC], ...
        'Color', [0.08,0.35,0.76], 'LineWidth', 2);
    plot([T.PL(lo);PLc], [T.rho(lo);rhoC], ...
        'Color', [0.82,0.17,0.11], 'LineWidth', 2);
    plot(PLc, rhoC, 'ko', 'MarkerFaceColor', 'k');
    yline(1, 'k--', 'rho=1', 'HandleVisibility', 'off');
    xline(PLc, 'k:', 'HandleVisibility', 'off');

    if j == 1
        set(gca, 'YScale', 'log');
        title('All equilibria: full spectral range');
    else
        ylim([0.72,1.6]);
        title('Zoom: high < 1, low > 1, critical = 1');
    end

    xlim([0,1.12*PLc]);
    xlabel('P_L (p.u.)');
    ylabel('rho(F_x)');
    grid on;
    box on;
    legend('high voltage branch', 'low voltage branch', ...
        'critical point', 'Location', 'best');
end

sgtitle('Figure 2: spectral radius at every enumerated positive equilibrium');
end
