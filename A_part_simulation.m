%% A部分：三节点ICM数值验证（MATLAB R2022b）

clearvars; clc;
outputDir=fullfile(fileparts(mfilename('fullpath')),'output','refactored');
if ~isfolder(outputDir), mkdir(outputDir); end
diary(fullfile(outputDir,'A_run.log'));
fprintf('A: original three-node simulation | MATLAB %s\n',version);

%% 三节点参数
cfg.three.u0 = 1;
cfg.three.g0L = 1;
cfg.three.g0G = 0.15;
cfg.three.gLG = 1.2;
cfg.three.PG = -0.3;
cfg.three.scanNumber = 201;
cfg.three.figureLoads = [0.45,0.90,1,1.10]; % (a)-(d)，只改变 PL。
cfg.three.searchEnabled = false; % 恢复并固定上一次参数。

cfg.three.searchG0L = [0.5,1,2];
cfg.three.searchG0G = [0.15,0.4,0.8];
cfg.three.searchGLG = [0.4,0.8,1.2];
cfg.three.searchPG = [-0.04,-0.12,-0.30,-0.60];

cfg.verbose = true;

%% Part 1. 三节点平方电压映射与完整正电压解集
% P=[PL;PG]，PG<0 表示 MPPT 恒功率源，未知平方电压 x=[vL^2;vG^2]。

p3=cfg.three;
parameterSearch=struct('enabled',false,'candidates',table(),'selectedIndex',NaN);
cfg.three=p3;
fprintf('Three-node parameters: g0L=%.4g, g0G=%.4g, gLG=%.4g, PG=%.4g.\n',...
    p3.g0L,p3.g0G,p3.gLG,p3.PG);
edges3 = [1,2,p3.g0L;1,3,p3.g0G;2,3,p3.gLG];
model3 = build_dc_model(3,edges3,1,p3.u0,zeros(3,1),[0;p3.PG]);
[PLcritical,uCritical] = three_critical_load(p3);
alpha3 = unique(sort([linspace(0.001,1.12,p3.scanNumber),p3.figureLoads]));
PLscan = PLcritical*alpha3;
records = zeros(0,11);  % [PL,vL,vG,rho,isHigh,residual,J11,J12,J21,J22,count]
rootCount = zeros(numel(PLscan),1);
allRoots = cell(numel(PLscan),1);
fprintf('PART 1 | n=3 | critical PL = %.12g\n',PLcritical);
for k=1:numel(PLscan)
    PL=PLscan(k); model3.P=[PL;p3.PG];
    U=three_all_positive_roots(p3,PL); allRoots{k}=U;
    rootCount(k)=size(U,2);
    if PL<PLcritical-1e-10
        assert(size(U,2)==2,'临界点以下应有两个正解。');
    elseif PL>PLcritical+1e-10
        assert(isempty(U),'临界点以上不应有正解。');
    else
        assert(size(U,2)==1,'临界点处应只有一个不同的正解。');
    end
    for j=1:size(U,2)
        x=U(:,j).^2; J=icm_jacobian(x,model3);
        rho=sparse_perron(J,x,model3.d);
        err=norm(x-icm_map(x,model3),inf);
        isHigh=(j==size(U,2)); % 消元变量 vG 严格递增，另做分量序检查。
        assert(err<2e-9);
        if isHigh
            assert(rho<=1+2e-7);
        else
            assert(all(U(:,j)<U(:,end)) && rho>1);
        end
        records(end+1,:)=[PL,U(:,j)',rho,isHigh,err,...
            J(1,1),J(1,2),J(2,1),J(2,2),size(U,2)]; 
        if cfg.verbose
            kind='low '; if isHigh, kind='high'; end
            fprintf('n=3 | PL=%.8f | %s | rho=%.10f | error=%.2e | PLcrit=%.10f\n',...
                PL,kind,rho,err,PLcritical);
        end
    end
    if isempty(U) && cfg.verbose
        fprintf('n=3 | PL=%.8f | no positive root | rho/error=N/A | PLcrit=%.10f\n',PL,PLcritical);
    end
end
model3.P=[PLcritical;p3.PG];
rhoCritical3=sparse_perron(icm_jacobian(uCritical.^2,model3),uCritical.^2,model3.d);
assert(abs(rhoCritical3-1)<1e-8);
equilibria3=array2table(records,'VariableNames',{'PL','vL','vG','rho',...
    'isHigh','residual','J11','J12','J21','J22','rootCount'});
figure1Audit=draw_figure1(p3,PLcritical,uCritical,cfg);
assert(isequal(figure1Audit.rootCount,[2,2,1,0]));
fprintf('Figure 1 contour checks: root counts=[%s], max normalized zero-level error=%.3e.\n',...
    num2str(figure1Audit.rootCount),figure1Audit.maxContourError);
draw_figure2(equilibria3,PLcritical,rhoCritical3);

%% 保存完整结果，导出图
resultsA=struct('parameters',p3,'PLcritical',PLcritical,...
    'uCritical',uCritical,'equilibria',equilibria3,'PLscan',PLscan,...
    'rootCount',rootCount,'allRoots',{allRoots},'parameterSearch',parameterSearch,...
    'figure1Audit',figure1Audit);
resultsA.checks=struct('positiveRootEnumeration',true,'originalPowerBalance',true,...
    'componentwiseHighSolution',true,'spectralSeparation',true,...
    'criticalRho',abs(rhoCritical3-1)<1e-8,'contourResidual',figure1Audit.maxContourError<2e-5);
save(fullfile(outputDir,'A_results.mat'),'resultsA','cfg');
writetable(equilibria3,fullfile(outputDir,'A_equilibria.csv'));
exportgraphics(figure(1),fullfile(outputDir,'Figure1.png'),'Resolution',180);
exportgraphics(figure(2),fullfile(outputDir,'Figure2.png'),'Resolution',180);
fprintf('A PASS | PLcritical=%.12g | critical rho=%.12g | max fixed-point residual=%.3e\n',...
    PLcritical,rhoCritical3,max(equilibria3.residual));
diary off;

%% ======================== LOCAL FUNCTIONS ==============================
function [R1,R2]=three_residual(vL,vG,s,PL)
% 真实三节点电阻网络节点功率平衡；适用于标量、向量和 contour 网格。
% 正号为消耗：CPL 的 PL>0，MPPT 的 PG<0。
% R_i = u_i*(YLL*u+YLS*uS)_i+P_i。
R1=(s.g0L+s.gLG)*vL.^2-s.gLG*vL.*vG-s.g0L*s.u0*vL+PL;
R2=(s.g0G+s.gLG)*vG.^2-s.gLG*vL.*vG-s.g0G*s.u0*vG+s.PG;
% 曲线形状的解析约束：
% R1=0 => vG=(a*vL-b+PL/vL)/c，二阶导数 2PL/(c*vL^3)>0。
% R2=0 => vL=(d*vG-t+PG/vG)/c，一阶导数 d/c-PG/(c*vG^2)>0。

end

function m=build_dc_model(n,edges,S,uS,Gbus,P)
% 由原始支路构建电导 Laplacian，然后在对角线加入并联阻性负载。
% 阻性负载功率是 G*u^2，已包含在 YLL 中。
ne=size(edges,1); S=S(:); L=setdiff((1:n)',S,'stable');
B=sparse([(1:ne)';(1:ne)'],[edges(:,1);edges(:,2)],...
    [ones(ne,1);-ones(ne,1)],ne,n);
Ynetwork=B'*spdiags(edges(:,3),0,ne,ne)*B;
Ybus=Ynetwork+spdiags(Gbus,0,n,n);
m.n=numel(L); m.total=n; m.S=S; m.L=L; m.uS=uS(:);
m.Ynetwork=Ynetwork; m.Ybus=Ybus;
m.YLL=Ybus(L,L); m.YLS=Ybus(L,S);
m.d=diag(m.YLL); m.Y0=spdiags(m.d,0,m.n,m.n);
m.Y1=m.YLL-m.Y0;
m.A=-spdiags(1./m.d,0,m.n,m.n)*m.Y1;
m.b=-(m.YLS*m.uS)./m.d; m.P=P(:);
assert(all(edges(:,3)>0) && all(m.d>0) && norm(Ynetwork*ones(n,1),inf)<1e-12);
assert(norm(Ybus-Ybus',inf)<1e-12);
adj=spones(-m.Y1); assert(max(conncomp(graph(adj)))==1,...
    '未知节点诱导子图不连通，不满足本代码采用的不可约假设。');
assert(all(m.b>=0) && any(m.b>0));
[~,flag]=chol(m.YLL); assert(flag==0,'YLL 必须为非奇异 M 矩阵。');
assert(issparse(Ybus) && issparse(m.Y0) && issparse(m.A));
end

function F=icm_map(x,m)
% 式 (46)：F=-Y0^{-1}(diag(sqrt(x))*Y1*sqrt(x)
%                         +diag(sqrt(x))*YLS*uS+P)。
% Y0 对角，因此 ./d 与左乘 Y0^{-1} 严格等价且不显式求逆。
assert(isreal(x) && all(isfinite(x)) && all(x>0),'映射输入必须属于 x>0 的正域。');
u=sqrt(x); F=-(u.*(m.Y1*u+m.YLS*m.uS)+m.P)./m.d;
end

function [J,W]=icm_jacobian(x,m)
% 式 (48)的 Jacobian。W=A o (sqrt(x)*sqrt(x)') 用两侧稀疏对角乘法实现。
u=sqrt(x); D=spdiags(u,0,m.n,m.n); W=D*m.A*D;
J=0.5*(W+spdiags(sum(W,2)+m.b.*u,0,m.n,m.n))*spdiags(1./x,0,m.n,m.n);
assert(issparse(J));
end

function [rho,eta,err]=sparse_perron(J,x,d)
% 数学定义 rho=max(abs(eig(J)))。对非负不可约矩阵求 Perron根，即最大模特征值，也就是论文中的

nn=size(J,1); scale=sqrt(d./x);
H=spdiags(scale,0,nn,nn)*J*spdiags(1./scale,0,nn,nn);
assert(norm(H-H',inf)<1e-10 && all(nonzeros(J)>=0));
H=(H+H')/2; % 只消除对称矩阵浮点舍入差异。
opts.tol=1e-12; opts.maxit=3000; opts.v0=ones(nn,1)/sqrt(nn);
[z,lambda,flag]=eigs(H,1,'largestreal',opts);
assert(flag==0,'Perron 特征值未收敛。');
rho=real(lambda(1,1)); eta=real(z)./scale;
if sum(eta)<0, eta=-eta; end
eta=eta/norm(eta,inf);
err=norm(J*eta-rho*eta,inf);
assert(all(eta>0) && err<1e-9*max(1,norm(J,inf)));
% 2x2 情形另用特征多项式独立核查全部特征值，不引入稠密矩阵。
if nn==2
    tr=J(1,1)+J(2,2); detJ=J(1,1)*J(2,2)-J(1,2)*J(2,1);
    ev=roots([1,-tr,detJ]);
    assert(abs(rho-max(abs(ev)))<1e-8*max(1,rho));
end
end

function [p,base]=three_polynomial(s,PL)
% 原始方程：R1=a*vL^2-c*vL*vG-b*vL+PL=0，
%            R2=d*vG^2-c*vL*vG-t*vG+PG=0。
% 消去 vL：w=vG>0，N=d*w^2-t*w+PG，vL=N/(c*w)。
% p(w)=c^2*w^2*N+b*c*w*N-a*N^2-PL*c^2*w^2。
% 消元没有平方操作；c*w>0 保证正域内等价，不遗漏任何正电压解。
a=s.g0L+s.gLG; d=s.g0G+s.gLG; c=s.gLG;
b=s.g0L*s.u0; t=s.g0G*s.u0; N=[d,-t,s.PG];
base=c^2*[N,0,0]+b*c*[0,N,0]-a*conv(N,N);
p=base-PL*c^2*[0,0,1,0,0];
end

function [PLcrit,uc]=three_critical_load(s)
% PL(w)=base(w)/(c^2*w^2)。其驻点满足 w*base'-2*base=0。
[~,base]=three_polynomial(s,0);
d=s.g0G+s.gLG; t=s.g0G*s.u0; c=s.gLG;
wmin=(t+sqrt(t^2-4*d*s.PG))/(2*d);
rr=roots(base.*[2,1,0,-1,-2]);
ws=real(rr(abs(imag(rr))<1e-9 & real(rr)>wmin));
assert(numel(ws)==1,'修改三节点参数后需重新核对所有物理驻点。');
w=ws(1); PLcrit=polyval(base,w)/(c^2*w^2);
uc=[(d*w^2-t*w+s.PG)/(c*w);w];
% PL(wmin)=0，PL(w)->-inf，唯一物理驻点给出全局最大可承载负荷。
assert(PLcrit>0 && all(uc>0));
end

function U=three_all_positive_roots(s,PL)
% 枚举四次多项式全部正域实根：导数全部实根分割单调区间，
% 在各区间检查符号变化，并在驻点直接检查偶重根（保留临界切点）。
% Cauchy 界覆盖所有根。
p=three_polynomial(s,PL); p=p/max(abs(p));
rr=roots(polyder(p)); cr=real(rr(abs(imag(rr))<1e-9));
d=s.g0G+s.gLG; t=s.g0G*s.u0; c=s.gLG;
wmin=(t+sqrt(t^2-4*d*s.PG))/(2*d);
bound=max(wmin+1,1+max(abs(p(2:end)/p(1))));
cuts=sort([wmin;cr(cr>wmin & cr<bound);bound]); candidates=[];
for k=1:numel(cuts)
    w=cuts(k);
    if abs(polyval(p,w))<=3e-13*max(1,polyval(abs(p),w))
        candidates(end+1)=w; 
    end
end
for k=1:numel(cuts)-1
    l=cuts(k); r=cuts(k+1); fl=polyval(p,l); fr=polyval(p,r);
    nearL=abs(fl)<=3e-13*max(1,polyval(abs(p),l));
    nearR=abs(fr)<=3e-13*max(1,polyval(abs(p),r));
    if ~nearL && ~nearR && sign(fl)~=sign(fr)
        candidates(end+1)=fzero(@(w) polyval(p,w),[l,r],...
            optimset('TolX',1e-13,'Display','off')); 
    end
end
U=zeros(2,0);
for w=sort(candidates)
    v=(d*w^2-t*w+s.PG)/(c*w);
    if v<=1e-9 || w<=0, continue; end
    [q1,q2]=three_residual(v,w,s,PL); q=[q1;q2];
    assert(norm(q,inf)<2e-9*max(1,PL),'消元根未通过原潮流方程检查。');
    if isempty(U) || all(max(abs(U-[v;w]),[],1)>1e-7*max(1,w))
        U(:,end+1)=[v;w]; 
    end
end
end

function audit=draw_figure1(s,PLc,uc,cfg)
% 将四个负荷工况叠加在同一坐标系，表示 PL 增加时 R1=0 的上移过程。
% 最初和最终 R1 曲线用实线，中间（包括临界负荷）用虚线；PG 固定，
% 因而 R2=0 只画一次。所有可见边界均直接使用 contour(...,R,[0 0])。
% 绿色另标低负荷下 R1<0 且 R2<0 的区域，严格对应 F(v.^2)>v.^2。
f=figure(1); clf(f);
set(f,'Name','Figure 1: CPL load increase in one plot','Color','w','Position',[40,50,1080,880]);
ax=axes('Parent',f); hold(ax,'on');
ratios=cfg.three.figureLoads;
assert(numel(ratios)==4 && all(diff(ratios)>0) && ratios(3)==1);
Ulow=three_all_positive_roots(s,ratios(1)*PLc);
vmax=1.18*max(Ulow(:));
audit.rootCount=zeros(1,4); audit.maxContourError=0;
audit.PL=ratios*PLc; audit.plotLimits=[0,vmax]; audit.voltageGaps=zeros(1,4);
allU=cell(1,4); rootsForGrid=uc;
for k=1:4
    U=three_all_positive_roots(s,audit.PL(k)); allU{k}=U;
    audit.rootCount(k)=size(U,2); rootsForGrid=[rootsForGrid,U]; %#ok<AGROW>
    if size(U,2)==2, audit.voltageGaps(k)=norm(U(:,2)-U(:,1)); end
end
% 在均匀网格上按原残差生成半透明掩码，image 的坐标与每个采样点一致。
% R1(PL_low)<=0 且 R1(PL_top)>=0 等价于位于两条 R1 零曲线之间。
vs=linspace(0.0001,vmax,1000); [VS,GS]=meshgrid(vs,vs);
[Rlow,Rfixed]=three_residual(VS,GS,s,audit.PL(1));
Rtop=three_residual(VS,GS,s,audit.PL(4));
band=(Rlow<=0 & Rtop>=0);
shadeTol=1e-11*max([1,abs(audit.PL(1)),abs(s.PG)]);
strict=(Rlow < -shadeTol & Rfixed < -shadeTol);
blue=[0.28,0.53,0.90]; green=[0.30,0.72,0.38];
image(ax,[vs(1),vs(end)],[vs(1),vs(end)],repmat(reshape(blue,1,1,3),numel(vs),numel(vs)),...
    'AlphaData',0.18*double(band),'HandleVisibility','off');
image(ax,[vs(1),vs(end)],[vs(1),vs(end)],repmat(reshape(green,1,1,3),numel(vs),numel(vs)),...
    'AlphaData',0.26*double(strict),'HandleVisibility','off');
set(ax,'YDir','normal');
hBand=patch(ax,NaN,NaN,blue,'FaceAlpha',0.18,'EdgeColor','none');
hStrict=patch(ax,NaN,NaN,green,'FaceAlpha',0.26,'EdgeColor','none');
audit.loadBandArea=nnz(band)*(vs(2)-vs(1))^2;
audit.lowLoadStrictArea=nnz(strict)*(vs(2)-vs(1))^2;
assert(audit.loadBandArea>0 && audit.lowLoadStrictArea>0);

% 将所有工况的已求平衡点加入 contour 网格，解析公式保持不变。
vx=unique(sort([linspace(0.0001,vmax,1000),rootsForGrid(1,:)]));
vy=unique(sort([linspace(0.0001,vmax,1000),rootsForGrid(2,:)]));
[VL,VG]=meshgrid(vx,vy);
lineStyles={'-','--','--','-'};
colors=[0.08,0.35,0.76;0.35,0.55,0.82;0.24,0.40,0.65;0.03,0.20,0.47];
hCPL=gobjects(1,4);
for k=1:4
    PL=audit.PL(k); U=allU{k};
    R1=three_residual(VL,VG,s,PL);
    [C1,hCPL(k)]=contour(ax,VL,VG,R1,[0,0],'Color',colors(k,:),...
        'LineWidth',2,'LineStyle',lineStyles{k});
    assert(strcmp(hCPL(k).Type,'contour'));
    audit.maxContourError=max(audit.maxContourError,contour_zero_error(C1,1,s,PL));
    if ~isempty(U)
        [rr1,rr2]=three_residual(U(1,:),U(2,:),s,PL);
        assert(max(abs([rr1,rr2]))<2e-9);
    end
end
[~,R2]=three_residual(VL,VG,s,audit.PL(1));
[C2,hMPPT]=contour(ax,VL,VG,R2,[0,0],'Color',[0.82,0.17,0.11],'LineWidth',2.3);
assert(strcmp(hMPPT.Type,'contour'));
audit.maxContourError=max(audit.maxContourError,contour_zero_error(C2,2,s,audit.PL(1)));
% contour 顶点来自网格插值；回代原残差核对每条曲线，并保留真实切点。
assert(audit.maxContourError<2e-5,'contour 零集未通过原残差回代检查。');
for k=1:3
    U=allU{k};
    if k==2
        plot(ax,U(1,:),U(2,:),'o','Color',colors(k,:),'MarkerFaceColor','w',...
            'MarkerSize',7,'LineWidth',1.5,'HandleVisibility','off');
    elseif k==3
        plot(ax,uc(1),uc(2),'kd','MarkerFaceColor',[0.95,0.65,0.12],...
            'MarkerSize',9,'HandleVisibility','off');
    else
        plot(ax,U(1,:),U(2,:),'ko','MarkerFaceColor','k','MarkerSize',7,'HandleVisibility','off');
    end
end
text(ax,Ulow(1,1)+0.025*vmax,Ulow(2,1)-0.045*vmax,'low voltage solution',...
    'FontSize',10,'BackgroundColor','w','Margin',1);
text(ax,Ulow(1,2)+0.025*vmax,Ulow(2,2)-0.055*vmax,'high voltage solution',...
    'HorizontalAlignment','left','FontSize',10,'BackgroundColor','w','Margin',1);
text(ax,uc(1)+0.015*vmax,uc(2)+0.12*vmax,'critical point',...
    'HorizontalAlignment','right','FontSize',11,'BackgroundColor','w','Margin',1);
% 上移方向只作注释：固定 vL 时 d(vG)/dPL=1/(gLG*vL)>0。
% 箭头端点取自最初、最终的真实 R1=0 方程，不对任何曲线作平移操作。
va=0.30*vmax; a=s.g0L+s.gLG; b=s.g0L*s.u0;
ya=(a*va-b+audit.PL(1)/va)/s.gLG;
yb=(a*va-b+audit.PL(4)/va)/s.gLG;
quiver(ax,va,ya+0.08*(yb-ya),0,0.80*(yb-ya),0,'Color',[0.20,0.27,0.40],...
    'LineWidth',1.5,'MaxHeadSize',0.25,'HandleVisibility','off');
text(ax,va+0.025*vmax,ya+0.25*(yb-ya),'P_L increases',...
    'Rotation',90,'FontSize',10,'Color',[0.20,0.27,0.40],'HorizontalAlignment','center');
labels={sprintf('R_1: %.2f P_{L,crit} (initial)',ratios(1)),...
    sprintf('R_1: %.2f P_{L,crit} (intermediate)',ratios(2)),...
    'R_1: P_{L,crit} (critical)',sprintf('R_1: %.2f P_{L,crit} (final, no roots)',ratios(4)),...
    'R_2: fixed P_G (MPPT)','Region swept by R_1','F(x)>x at the initial load'};
legend(ax,[hCPL,hMPPT,hBand,hStrict],labels,'Location','southeast','FontSize',9);
xlim(ax,[0,vmax]); ylim(ax,[0,vmax]); axis(ax,'square'); grid(ax,'on'); box(ax,'on');
set(ax,'FontSize',11,'Layer','top');
xlabel(ax,'v_L (p.u.)'); ylabel(ax,'v_G (p.u.)');
assert(audit.voltageGaps(2)<audit.voltageGaps(1));
title(ax,{'Figure 1: increasing CPL load - two equilibria, merging, then no equilibrium',...
    sprintf('g_{0L}=%.2g, g_{0G}=%.2g, g_{LG}=%.2g, fixed P_G=%.2g; P_{L,crit}=%.8f',...
    s.g0L,s.g0G,s.gLG,s.PG,PLc)});
end

function err=contour_zero_error(C,which,s,PL)
% MATLAB contour 矩阵由 [等值线值;顶点数] 头和对应顶点逐段组成。
% 回代实际潮流残差检查每一条 contour，而不是目测曲线形状。
err=0; j=1; total=0;
while j<size(C,2)
    count=C(2,j); V=C(:,j+1:j+count);
    [r1,r2]=three_residual(V(1,:),V(2,:),s,PL);
    if which==1, r=r1; else, r=r2; end
    scale=max([1,abs(PL),abs(s.PG)]);
    err=max(err,max(abs(r))/scale); total=total+count;
    j=j+count+1;
end
assert(total>0,'指定正电压范围内没有找到残差零等值线。');
end

function draw_figure2(T,PLc,rhoC)
f=figure(2); clf(f); set(f,'Name','Figure 2: Three-node spectral radius','Color','w','Position',[80,80,1280,550]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
hi=T.isHigh==1 & T.PL<PLc-1e-10; lo=T.isHigh==0;
for j=1:2
    nexttile; hold on;
    % 两条曲线均包含同一个已求出的临界点，明确显示支路在 rho=1 合并。
    plot([T.PL(hi);PLc],[T.rho(hi);rhoC],'Color',[0.08,0.35,0.76],'LineWidth',2);
    plot([T.PL(lo);PLc],[T.rho(lo);rhoC],'Color',[0.82,0.17,0.11],'LineWidth',2);
    plot(PLc,rhoC,'ko','MarkerFaceColor','k');
    yline(1,'k--','rho=1','HandleVisibility','off'); xline(PLc,'k:','HandleVisibility','off');
    if j==1
        set(gca,'YScale','log'); title('All equilibria: full spectral range');
    else
        ylim([0.72,1.6]); title('Zoom: high < 1, low > 1, critical = 1');
    end
    xlim([0,1.12*PLc]); xlabel('P_L (p.u.)'); ylabel('rho(F_x)'); grid on; box on;
    legend('high voltage branch','low voltage branch','critical point','Location','best');
end
sgtitle('Figure 2: spectral radius at every enumerated positive equilibrium');
end
