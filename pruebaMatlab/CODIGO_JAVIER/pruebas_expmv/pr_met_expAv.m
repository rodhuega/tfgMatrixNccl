function pr_met_expAv(f,dat,flag_figure,pr)
% f: funciones a comparar
% dat: argumento 2º de la función (cosher(A,7))
%     si no tiene argumentos la función, su valor es -1
% flag_figure: si vale 0 obtiene fig, si no obtiene eps
% pr: tipo de prueba (1, 2 o 3)
%f{1}='expmvtayn';f{2}='expAv_taylor';f{3}='expmvm';dat=[30 4 0];pr_met_expAv(f,dat,0,4)

switch pr
    case 1
        test='expmv_diag_hadamard_complex_n128_nd256';nmat=100;
        [T,E,cond,prods]=calcula_errores(test,f,dat,nmat);
	case 2
        test='expmv_jordan_hadamard_complex_n128_boundvp10_maxmult5_nd256';nmat=100;
        [T,E,cond,prods]=calcula_errores(test,f,dat,nmat);
	case 3
        test='expmv_toolbox_n128_nd256';nmat=52;
        [T,E,cond,prods]=calcula_errores(test,f,dat,nmat);  
    case 4
        test='expmv_eigtool_n128_nd256';nmat=20;
        [T,E,cond,prods]=calcula_errores(test,f,dat,nmat);    
end
make_graph(test,f,T,E,cond,prods,14,flag_figure)
end

function [T,E,cond,Np]=calcula_errores(test,f,dat,nmat)
 tmat='FA';
    
nf=length(f);T=zeros(nf,nmat);Np=zeros(nf,nmat);
k=0;
for i=1:nmat
    %fprintf("i=%d\n",i);
    t=sprintf('load %s_%s',test,int2str(i));
    eval(t);
    %eval(['load ',test,'_',int2str(i)]);
    tc=sprintf('max(max(isnan(%s)))||max(max(isinf(%s))||max(isinf(condFA)))',tmat,tmat);
%    if max(max(isnan(expA)))||max(max(isinf(expA)))
    if eval(tc) 
        continue
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    try
        if dat(1)>=0
            tic
            [C,m(1),s(1),np] = feval(f{1},A,v,dat(1));
            t=toc;
        else
            tic
            [C,m(1),s(1),np] = feval(f{1},A,v);
            t=toc;
        end
    catch
        continue     
    end
    tc=sprintf('Er=norm(%s-C)/norm(%s);',tmat,tmat);
    %tc=sprintf('Er=norm(%s-C,1)/norm(%s,1);',tmat,tmat);
    eval(tc);
	if Er>1e-2%Esto lo hago para evitar casos de mucho error
        continue
    end
    k=k+1;
    cond(k)=condFA;
    T(1,k)=t;
    E(1,k)=Er;
    Np(1,k)=np;
    for j=2:nf
        if dat(j)>=0
            tic
            [C,m(j),s(j),np] = feval(f{j},A,v,dat(j));
            t=toc;
        else
            tic
            [C,m(j),s(j),np] = feval(f{j},A,v);
            t=toc;
        end
        cond(k)=condFA;
        T(j,k)=t;
        %tc=sprintf('E(j,k)=norm(%s-C)/norm(%s);',tmat,tmat);
        tc=sprintf('E(j,k)=norm(%s-C,1)/norm(%s,1);',tmat,tmat);
        eval(tc);
        Np(j,k)=np;
    end
%         fprintf('k=%d: Error_%s(m=%d,s=%d,np=%d): %g\n',k,f{1},m(1),s(1),Np(1,k),E(1,k));
%         fprintf('k=%d: Error_%s(m=%d,s=%d,np=%d): %g\n',k,f{2},m(2),s(2),Np(2,k),E(2,k)); 
%         fprintf('k=%d: Error_%s(m=%d,s=%d,np=%d): %g\n\n',k,f{3},m(3),s(3),Np(3,k),E(3,k));
end
T=T(1:nf,1:k);
Np=Np(1:nf,1:k);
end


function make_graph(test,f,T,E,cond,prods,fontsize,flag_figure)
try
    warning off
	mkdir Figures
catch
end
[np,nmet]=size(T);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Normwise
figure(1)
[condAm, J]=sort(cond, 'descend');  % Con la 1 norma
nmet=size(E,1);
np=size(E,2);
for i=1:nmet
	for j=1:np
        Em(i,j)=E(i,J(j));
    end
end
    
sim{1}='+k'; sim{2}='xk'; sim{3}='sk'; sim{4}='vk'; sim{5}='*k'; sim{6}='ok'; sim{7}='dk'; sim{8}='pk'; sim{9}='hk';
t='semilogy(1:np,condAm*2^(-53),''k''';
for i=1:nmet
	t=[t,',1:np,Em(',int2str(i),',:),''',sim{i},''''];
end
t=[t,')'];
eval(t);

%%%%%Leyenda
leyenda_aux=['''',f{1},''''];
for j=2:nmet
     leyenda_aux=[leyenda_aux,',','''',f{j},''''];
end
eval(['legend(''cond*u''',',',leyenda_aux,');']);
set(gca,'FontSize',fontsize);

xlabel('Matrix')
ylabel('Er')
%aux=['Figures\Fig_',test,'_normwise_',f1,'_',f2,'_',f3];
aux=['Figures\Fig_',test,'_normwise'];
nfun=[];
for j=1:nmet
    nfun=[nfun,'_',f{j}];
end
aux=[aux,nfun];
%%%%%%%%%%%%
%axis([0 np 0 max(max([E;condAm*2^(-53)]))]);
if flag_figure
    %saveas(gcf,aux, 'jpg');
    % Esto hace que el eps imprima el fondo blanco y no gris
    set(gcf, 'Color', [1,1,1]);
    saveas(gcf,aux, 'epsc'); % Changed for generate epsc files
else
    saveas(gcf,aux, 'fig');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Performance profile
figure(2)
alpha=1:0.1:5;
lalpha=length(alpha);
P=zeros(nmet,lalpha);
for j=1:np
    minE=min(E(:,j));
	for k=1:nmet    
        for i=1:lalpha
            if E(k,j)<=alpha(i)*minE
                P(k,i)=P(k,i)+1;
            end
        end
	end
end
np=size(E,2);
P=P/np;
sim{1}='-+k'; sim{2}='-xk'; sim{3}='-sk'; sim{4}='-vk'; sim{5}='-*k'; sim{6}='-ok'; sim{7}='-dk'; sim{8}='-pk'; sim{9}='-hk';
t='plot(alpha,P(1,:),''-+k''';
for i=2:nmet
    t=[t,',alpha,P(',int2str(i),',:),''',sim{i},''''];
end
t=[t,')'];
eval(t);
%%%%%%%%%%%%
%axis([0 np 0 max(max(P))]);
%%%%%%%%%%%%%%%%%%%%%%%%%
set(gca,'FontSize',fontsize);
%%%%%Leyenda
eval(['legend(',leyenda_aux,');']);

xlabel('\it \alpha')
ylabel('p')
aux=['Figures\Fig_',test,'_nprofile_'];
aux=[aux,nfun];
if flag_figure
    %saveas(gcf,aux, 'jpg');
    % Esto hace que el eps imprima el fondo blanco y no gris
    set(gcf, 'Color', [1,1,1]);
    saveas(gcf,aux, 'epsc'); % Changed for generate epsc files
else
    saveas(gcf,aux, 'fig');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Errores relativos
figure(3)
for k=1:nmet-1
    Er(k,:)=E(k,:)./E(nmet,:);
end
[Erm0, J]=sort(Er(1,:), 'descend');  % Con la 1 norma
for i=1:nmet-1
	for j=1:np
        Erm(i,j)=Er(i,J(j));
    end
end
sim{1}='+k'; sim{2}='xk'; sim{3}='sk'; sim{4}='vk'; sim{5}='*k'; sim{6}='ok'; sim{7}='dk'; sim{8}='pk'; sim{9}='hk';
t=['semilogy(1:np,Erm(',int2str(1),',:),''',sim{1},''''];
for i=2:nmet-1
    t=[t,',1:np,Erm(',int2str(i),',:),''',sim{i},''''];
end
t=[t,')'];
eval(t);

t='legend(';
for j=2:nmet
    G=['E(',f{1},')/E(',f{j},')'];
    t=[t,'''',G,''''];
    if j<nmet
        t=[t,','];
    else
        t=[t,')'];
    end
end
eval(t);

set(gca,'FontSize',fontsize);
grid
hold on

% X=1:np;
% plot(X,ones(1,np),'k');
% plot(X,0.5*ones(1,np),'k');
xlabel('Matrix')
ylabel('Ratio relative errors')
aux=['Figures\Fig_',test,'_ratio_errors'];
aux=[aux,nfun];
if flag_figure
    %saveas(gcf,aux, 'jpg');
    % Esto hace que el eps imprima el fondo blanco y no gris
    set(gcf, 'Color', [1,1,1]);
    saveas(gcf,aux, 'epsc'); % Changed for generate epsc files
else
    saveas(gcf,aux, 'fig');
end
hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Número de productos relativo
figure(4)
for k=1:nmet-1
    Er(k,:)=prods(1,:)./prods(k+1,:);
end
[Erm0, J]=sort(Er(1,:), 'descend');  % Con la 1 norma
for i=1:nmet-1
	for j=1:np
        Erm(i,j)=Er(i,J(j));
    end
end
sim{1}='+k'; sim{2}='xk'; sim{3}='sk'; sim{4}='vk'; sim{5}='*k'; sim{6}='ok'; sim{7}='dk'; sim{8}='pk'; sim{9}='hk';
t=['semilogy(1:np,Erm(',int2str(1),',:),''',sim{1},''''];
for i=2:nmet-1
    t=[t,',1:np,Erm(',int2str(i),',:),''',sim{i},''''];
end
t=[t,')'];
eval(t);

t='legend(';
for j=2:nmet
    G=['M(',f{1},')/M(',f{j},')'];
    t=[t,'''',G,''''];
    if j<nmet
        t=[t,','];
    else
        t=[t,')'];
    end
end
eval(t);
%axis([0 np min(min(Erm)) max(max(Erm))])
set(gca,'FontSize',fontsize);
grid
hold on

% X=1:np;
% plot(X,ones(1,np),'k');
% plot(X,0.5*ones(1,np),'k');
xlabel('Matrix')
ylabel('Ratio matrix products')
aux=['Figures\Fig_',test,'_ratio_matrix_products'];
aux=[aux,nfun];
if flag_figure
    %saveas(gcf,aux, 'jpg');
    % Esto hace que el eps imprima el fondo blanco y no gris
    set(gcf, 'Color', [1,1,1]);
    saveas(gcf,aux, 'epsc'); % Changed for generate epsc files
else
    saveas(gcf,aux, 'fig');
end
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nComparativa  para %d matrices del test %s:\n',np,test);
for j=2:nmet
    comparam([E(1,:);E(j,:)],f{1},f{j});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Tiempo total:\n');
for j=1:nmet
    fprintf('\t%15s: %f\n',f{j},sum(T(j,:)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Nº productos total:\n');
for j=1:nmet
    fprintf('\t%15s: %d\n',f{j},sum(prods(j,:)));
end
close all
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function compara(E,f1,f2)
np=size(E,2);

alfa=[0.1 0.2 0.5 1 2 5 10]; 
lalfa=length(alfa);
I=zeros(1,lalfa+1);

for i=1:np
    if E(2,i)<E(1,i)*alfa(1)
        I(1)=I(1)+1;
    end        
    for j=1:lalfa-1
        if E(1,i)*alfa(j)<=E(2,i)&&E(2,i)<E(1,i)*alfa(j+1)
            I(j+1)=I(j+1)+1;
        end
    end
    if E(2,i)>=E(1,i)*alfa(lalfa)
        I(lalfa+1)=I(lalfa+1)+1;
    end
end
if sum(I)~=np
    error('No está bien calculado')
end
I=I/np*100;
fprintf('E(%s)<%.1f*E(%s):%.2f\n',f2,alfa(1),f1,I(1));
for j=1:lalfa-1
    fprintf('%.1f*E(%s)<=E(%s)<%.1f*E(%s):%.2f\n',alfa(j),f1,f2,alfa(j+1),f1,I(j+1));
end
fprintf('%.1f*E(%s)<=E(%s):%.2f\n\n',alfa(lalfa),f1,f2,I(lalfa));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function comparam(E,f1,f2)
np=size(E,2);
menor=0;mayor=0;igual=0;
for i=1:np
    if E(1,i)<E(2,i)
        menor=menor+1;
    elseif E(1,i)>E(2,i)
        mayor=mayor+1; 
    else
        igual=igual+1;
    end
end
fprintf('E(%s)<E(%s) en %.2f%% ocasiones\n',f1,f2,menor/np*100);
fprintf('E(%s)>E(%s) en %.2f%% ocasiones\n',f1,f2,mayor/np*100);
fprintf('E(%s)=E(%s) en %.2f%% ocasiones\n\n',f1,f2,igual/np*100);
end
