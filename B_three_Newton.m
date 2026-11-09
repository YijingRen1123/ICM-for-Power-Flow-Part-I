% 123-node DC microgrid:
% 10 voltage sources + 10 MPPT sources + 103 CPLs
clear; clc;

%% Admittance matrix Y, YLS, YLL, YSL, YSS, Y0, Y1
n=123; S=(1:10)'; L=(11:123)'; m=numel(L);

% Upper-triangular branch conductances
Y=zeros(123,123);
Y(1,23)=1;
Y(23,24)=1;
Y(21,22)=0.5;
Y(22,24)=0.5;
Y(24,25)=1;
Y(25,26)=1;
Y(26,29)=1;
Y(28,29)=1;
Y(27,28)=1;
Y(11,27)=0.2;
Y(29,34)=0.5;
Y(2,34)=0.25;
Y(34,35)=0.25;
Y(35,44)=0.1;
Y(43,44)=0.2;
Y(16,43)=0.2;
Y(44,57)=0.5;
Y(35,36)=0.2;
Y(44,45)=0.2;
Y(45,65)=0.5;
Y(64,65)=0.2;
Y(65,66)=1;
Y(15,66)=0.25;
Y(42,45)=0.5;
Y(40,42)=0.2;
Y(40,41)=0.5;
Y(37,40)=0.5;
Y(37,38)=0.25;
Y(38,39)=0.2;
Y(31,37)=0.5;
Y(30,31)=1;
Y(31,32)=0.2;
Y(32,33)=0.5;
Y(12,33)=0.2;
Y(14,42)=0.2;
Y(57,58)=0.5;
Y(58,59)=0.5;
Y(4,59)=0.25;
Y(59,63)=0.5;
Y(55,57)=0.2;
Y(55,56)=0.5;
Y(51,55)=1;
Y(48,51)=1;
Y(47,48)=1;
Y(46,47)=1;
Y(3,46)=0.5;
Y(46,49)=1;
Y(54,55)=1;
Y(53,54)=1;
Y(50,53)=1;
Y(52,53)=1;
Y(53,60)=0.2;
Y(5,60)=1;
Y(60,61)=0.5;
Y(61,62)=0.5;
Y(57,74)=0.1;
Y(74,75)=0.2;
Y(75,76)=0.2;
Y(76,77)=0.5;
Y(19,77)=0.5;
Y(68,76)=0.1;
Y(67,68)=0.25;
Y(17,67)=0.25;
Y(68,69)=0.2;
Y(69,89)=0.5;
Y(89,90)=0.2;
Y(69,70)=0.5;
Y(69,92)=0.1;
Y(92,93)=0.5;
Y(93,94)=0.5;
Y(94,95)=0.5;
Y(13,95)=0.25;
Y(71,72)=0.5;
Y(70,71)=0.5;
Y(72,73)=0.5;
Y(72,108)=0.25;
Y(73,112)=0.5;
Y(112,113)=0.5;
Y(8,113)=0.2;
Y(108,109)=0.5;
Y(110,111)=0.5;
Y(105,108)=0.5;
Y(105,106)=0.5;
Y(106,107)=0.5;
Y(7,107)=0.5;
Y(102,105)=0.5;
Y(102,103)=0.5;
Y(103,104)=0.5;
Y(101,102)=0.5;
Y(100,101)=0.5;
Y(96,100)=0.5;
Y(97,100)=0.5;
Y(96,97)=0.5;
Y(97,98)=0.5;
Y(98,99)=0.5;
Y(6,99)=0.2;
Y(109,110)=0.25;
Y(72,91)=0.25;
Y(91,114)=0.5;
Y(114,115)=0.5;
Y(115,116)=0.5;
Y(9,116)=0.2;
Y(88,91)=0.1;
Y(88,119)=1;
Y(87,88)=0.1;
Y(86,87)=0.1;
Y(85,86)=0.2;
Y(84,86)=0.25;
Y(83,84)=0.2;
Y(82,84)=0.25;
Y(81,82)=0.2;
Y(80,82)=0.25;
Y(79,80)=0.2;
Y(78,80)=0.25;
Y(20,78)=0.2;
Y(118,119)=1;
Y(18,118)=0.5;
Y(118,120)=0.5;
Y(120,122)=0.5;
Y(121,122)=0.2;
Y(122,123)=0.2;
Y(10,123)=0.2;
Y(117,121)=0.2;

% Construct the nodal admittance matrix
Y=Y+Y';
Y=diag(Y*ones(123,1))-Y;
%Branch=find(triu(Y,1)); 
Y=sparse(Y);

% Matrix partition
YSS=Y(S,S);
YSL=Y(S,L);
YLS=Y(L,S);
YLL=Y(L,L);

d=diag(YLL);
Y0=eye(113).*YLL;
Y1=YLL-Y0;

%% Load vector P and voltage reference uS
% DC power-flow equation: diag(uL)*(YLL*uL+YLS*uS)+P=0
% Positive P: CPL consumption; Negative P: MPPT power injection

uS = ones(10,1);
PG = -0.11*ones(10,1);      % 11–20
PL0 = 0.0063*ones(103,1);   % 21–123
P  = [PG; PL0];
A=YLL^(-1)*diag(P);

%Fractional Newton
% uL = 1-A*uL^(-1), A = YLL^(-1)*diag(P)
x=ones(113,21);
for i=1:20
x(:,i+1)=x(:,i)-(eye(113)-A*diag(x(:,i))^(-2))^(-1)*(x(:,i)+YLL^(-1)*(P./x(:,i))-ones(113,1));
end

%Quadratic Newton
% f(uL)=diag(uL)*(YLL*uL+YLS*uS)+P=0
% F(uL)=df/duL: Jacobian matrix
y=ones(113,11);
for i=1:10
   F=diag(YLL*y(:,i)+YLS*uS)+diag(y(:,i))*YLL;  %Jacobian matrix
   f=diag(y(:,i))*(YLL*y(:,i)+YLS*uS)+P;        %f(x) equation
   y(:,i+1)=y(:,i)-F^(-1)*f;
end 

%ICM Newton
% Increasing-concave mapping: 
% x = uL.^2, and x = F(x)
% J = dF(x)/dx: Analytical Jacobian, 
% Newton iteration: x(k+1)=x(k)-(I-J)^(-1)*(x(k)-F(x(k)))
z=ones(113,11);
for i=1:10
   F=-Y0^(-1)*(diag(sqrt(z(:,i)))*Y1*sqrt(z(:,i))+diag(sqrt(z(:,i)))*YLS*uS+P);
   W=diag(sqrt(z(:,i)))*(-Y0^(-1)*Y1)*diag(sqrt(z(:,i)));
   J=0.5*(W+diag(sum(W,2)-Y0^(-1)*YLS*uS.*sqrt(z(:,i))))*diag(1./z(:,i));
   z(:,i+1)=z(:,i)-(eye(113)-J)^(-1)*(z(:,i)-F);
end

% Plot Comparison of Three Methods
figure;
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
data={x,y,sqrt(z)};
names={'Fractional Newton','Quadratic Newton','ICM Newton'};
C=lines(113);

for j=1:3
    nexttile; hold on;
    for k=1:113
        plot(0,data{j}(k,1),'*','Color',C(k,:),'MarkerSize',4);
        plot(1:size(data{j},2)-1,data{j}(k,2:end),...
            '--*','Color',C(k,:),'LineWidth',0.8,'MarkerSize',4);
    end
    xlabel('Iteration k'); ylabel('Node voltage (p.u.)');
    title(names{j}); grid on; box on;
end



%% Figure : Spectral radius of high-voltage equilibrium
% CPL load continuation:
% P(alpha)=[PG; alpha*PL0], MPPT power remains unchanged
% Original power-flow equation:
% R(uL,alpha)=diag(uL)*(YLL*uL+YLS*uS)+P(alpha)=0
Pfun = @(a) [PG;a*PL0];
R = @(u,a) u.*(YLL*u+YLS*uS)+Pfun(a);

% Increasing-concave mapping parameters
% Aicm=-Y0^(-1)*Y1, b=-Y0^(-1)*YLS*uS

Aicm = -Y0\Y1;
b = -Y0\(YLS*uS);

opts = optimoptions('fsolve','Display','off',...
    'FunctionTolerance',1e-11,'StepTolerance',1e-11);

% Initial equilibrium
u = ones(113,1);
[u,~,~] = fsolve(@(u) R(u,0),u,opts);

% Find critical loading factor
alpha = 0;
da = 0.01;

x0 = u.^2;
D = diag(sqrt(x0));
W = D*Aicm*D;
J = 0.5*(W+diag(sum(W,2)+b.*sqrt(x0)))*diag(1./x0);
rho = max(abs(eig(J)));

while rho < 1
    alpha0 = alpha;
    rho0 = rho;

    alpha = alpha+da;
    [unew,~,flag] = fsolve(@(u) R(u,alpha),u,opts);

    if flag<=0 || any(unew<=0)
        alpha = alpha0;
        da = da/2;
        if da<1e-7, break; end
        continue;
    end

    x0 = unew.^2;
    D = diag(sqrt(x0));
    W = D*Aicm*D;
    J = 0.5*(W+diag(sum(W,2)+b.*sqrt(x0)))*diag(1./x0);
    rho = max(abs(eig(J)));

    if rho>=1
        alpha_c = alpha0+(1-rho0)*(alpha-alpha0)/(rho-rho0);
        break;
    end

    u = unew;
end

% Sampling points
alphaGrid = unique(sort([0,0.1:0.05:0.98*alpha_c,...
    alpha_c*[0.98,0.99,0.995,0.999],alpha_c]));

rhoData = zeros(size(alphaGrid));
u = ones(113,1);

for k = 1:length(alphaGrid)

    a = alphaGrid(k);

    if k==length(alphaGrid)
        rhoData(k)=1;
        continue;
    end

    [u,~,~] = fsolve(@(u) R(u,a),u,opts);

    x0 = u.^2;
    D = diag(sqrt(x0));
    W = D*Aicm*D;
    J = 0.5*(W+diag(sum(W,2)+b.*sqrt(x0)))*diag(1./x0);

    rhoData(k)=max(abs(eig(J)));
end

%% Plot
figure(3); clf;
set(gcf,'Color','w','Position',[110,80,1050,650]);

plot(alphaGrid,rhoData,'o-','Color',[0.08,0.35,0.76],...
    'LineWidth',1.7,'MarkerSize',4);
hold on;

plot(alpha_c,1,'ko','MarkerFaceColor','k','MarkerSize',7);
yline(1,'k--','\rho=1','LabelHorizontalAlignment','left');
xline(alpha_c,'k:');

text(0.04,0.88,...
    sprintf('\\rho=1  Critical load: \\alpha_c=%.8f',alpha_c),...
    'Units','normalized');

xlabel('Load factor \alpha');
ylabel('\rho(F_x(x_h,\alpha))');

title({'Spectral radius of the high-voltage solution',...
    '10 voltage sources, 10 MPPT sources, 103 CPLs'});

legend('High-voltage branch','Critical point',...
    'Location','southeast');

xlim([0,1.2]);
ylim([0.996,1.0005]);
yticks(0.996:0.0005:1.0005);

set(gca,'FontName','Times New Roman','FontSize',11);
grid on; box on;



