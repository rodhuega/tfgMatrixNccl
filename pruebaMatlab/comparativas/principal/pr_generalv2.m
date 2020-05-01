function pr_generalv2(test,f,dat,flag_figure)
% pr_generalv2(test,f,dat,flag_figure)
%
% Función que compara resultados entre diferentes tipos de métodos de 
% cálculo de funciones matriciales.
%
% Datos de entrada:
% - test: Array de celdas con el nombre del directorio donde residen las 
%         matrices de test a emplear en la comparativa.
% - f:    Array de celdas con el nombre de los métodos de cálculo a 
%         comparar.
% - dat:  Vector, de longitud igual a los arrays previos, con el segundo
%         parámetro de entrada de la función a invocar para cada uno de los
%         métodos anteriores. Si la función no tiene tal argumento, su
%         valor será -1.
% - flag_figure: Formato de las gráficas generadas: 0 (fig), otro (eps).
%
% Ejemplo de invocación para la exponencial
% test{1}='exp_diag_hadamard_complex_n128_nd256';
% test{2}='exp_jordan_hadamard_complex_n128_boundvp10_maxmult5_nd256';
% test{3}='exp_toolbox_n128_nd256';
% test{4}='exp_eigtool_n128_nd256';
% f{1}='expmber';f{2}='expmtay';f{3}='exptaynsv3';f{4}='expm_newm';
% dat=[1 1 9 0];flag_figure=0;
% pr_generalv2(test,f,dat,flag_figure)

% Ejemplo de invocación para el coseno
% test{1}='cos_diag_hadamard_complex_n128_nd256';
% test{2}='cos_jordan_hadamard_complex_n128_boundvp10_maxmult5_nd256';
% test{3}='cos_toolbox_n128_nd256';
% test{4}='cos_eigtool_n128_nd256';
% f{1}='cosmber';f{2}='cosmtay';f{3}='cosmtayher';f{4}='cosm';
% dat=[1 1 -1 0];flag_figure=0;
% pr_generalv2(test,f,dat,flag_figure)

%Versión v2 (17-7-2019)
% Se ha modificado la versión anterior (pr_generalv1), de manera que 
% en las gráficas normwise no aparezcan las matrices con condFA inf o nan
% Se ha invertido el cociente del ratio de productos.

T=[];E=[];prods=[];cond=[];M=[];S=[];
t=test{1};
for i=1:length(test)
    if i>1
        t=[t,'-',test{i}];
    end
    eval(test{i});
	[T1,E1,cond1,prods1,M1,S1]=calcula_errores(test{i},f,dat,nmat); 
	T=[T,T1];E=[E,E1];prods=[prods,prods1];cond=[cond,cond1];M=[M,M1];S=[S,S1];
end

make_graph(t,f,T,E,cond,prods,M,S,14,flag_figure)
end

function [T,E,cond,Np,M,S]=calcula_errores(test,f,dat,nmat)
fprintf('Test %s:\n',test); 
nf=length(f);T=zeros(nf,nmat);Np=zeros(nf,nmat);cond=zeros(1,nmat);M=zeros(nf,nmat);S=zeros(nf,nmat);
m=zeros(1,nf);s=zeros(1,nf);np=zeros(1,nf);
Er=zeros(1,nf);t=zeros(1,nf);
k=0;
for i=1:nmat
    flag_error=0;
    eval(['load ',test,'_',int2str(i)]);
    if flag_error
        fprintf('\tLa matriz %d no está calculada\n',i);
        continue
    end
    if isinf(condFA) || isnan(condFA) || (eps/2*condFA>=1000)
        fprintf('\tLa matriz %d no formará parte de la gráfica normwise (condFA=%e)\n',i,condFA);
    end    
    for j=1:nf
        try
            if dat(j)>=0
                tic
                [C,m(j),s(j),np(j)]= feval(f{j},A,dat(j));
                t(j)=toc;
            else
                tic
                [C,m(j),s(j),np(j)]= feval(f{j},A);
                t(j)=toc;
            end
            Er(j)=double(norm(FA-C,1)/norm(FA,1));
            if Er(j)>=10 || isnan(Er(j)) % Si el error relativo es demasiado grande, 
                                         % prescindimos de la matriz                         
                flag_error=1;
                fprintf('\tLa matriz %d no se considerará. %s: error relativo=%.3e\n',i, f{j},Er(j));
            end
        catch
            flag_error=1;
            fprintf('\tLa matriz %d no se considerará. Ha habido un error al ejecutar %s\n',i,f{j})
        end
    end
    if flag_error
        continue
    else
        k=k+1;
        cond(k)=condFA;
        for j=1:nf
            T(j,k)=t(j);   
            Np(j,k)=np(j);
            E(j,k)=Er(j);
            M(j,k)=m(j);
            S(j,k)=s(j);
        end
    end
end
fprintf('\tEmpleadas con éxito %d de %d matrices\n',k,nmat);
T=T(1:nf,1:k);
Np=Np(1:nf,1:k);
cond=cond(1:k);
M=M(1:nf,1:k);
S=S(1:nf,1:k);
end

function make_graph(test,f,T,E,cond,prods,M,S,fontsize,flag_figure)
try
    warning off
	mkdir Figures
catch
end

F = strrep(f,'_','\_');

%[np,nmet]=size(T);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Normwise
figure(1)
[condF, J]=sort(cond, 'descend');  % Con la 1 norma
nmet=size(E,1);
np=size(E,2);
for i=1:nmet
	for j=1:np
        Em(i,j)=E(i,J(j));
    end
end

for i=1:np
    if (~isinf(condF(i)))&&(~isnan(condF(i)))&&(eps/2*condF(i)<1000)
        condFA=eps/2*condF(i:np);
        Ef=Em(:,i:np);
        np=np-i+1;
        break
    end
end

sim{1}='+k'; sim{2}='xk'; sim{3}='sk'; sim{4}='vk'; sim{5}='*k'; sim{6}='ok'; sim{7}='dk'; sim{8}='pk'; sim{9}='hk';
t='semilogy(1:np,condFA,''k''';
for i=1:nmet
	t=[t,',1:np,Ef(',int2str(i),',:),''',sim{i},''''];
end
t=[t,')'];
eval(t);

%%%%%Leyenda
leyenda_aux=['''',F{1},''''];
for j=2:nmet
     leyenda_aux=[leyenda_aux,',','''',F{j},''''];
end
eval(['legend(''cond*u''',',',leyenda_aux,');']);
set(gca,'FontSize',fontsize);

xlabel('Matrix')
ylabel('Er')
%axis([0 300 10^-18 10^2]); 
%axis([0 300 10^-20 1]); 
% Comentado para cambiar el nombre al fichero de la gráfica
%aux=['Figures\Fig_',test,'_normwise'];
aux=['Figures\normwise_',test];
nfun=[];
%for j=1:nmet
for j=1:1
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

% Comentado para cambiar el nombre al fichero de la gráfica
%aux=['Figures\Fig_',test,'_nprofile_'];
aux=['Figures\nprofile_',test];
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
    G=['E(',F{1},')/E(',F{j},')'];
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
ylabel('Relative error ratio')

% Comentado para cambiar el nombre al fichero de la gráfica
%aux=['Figures\Fig_',test,'_error_ratio'];
aux=['Figures\error_ratio_',test];
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
    %Er(k,:)=prods(1,:)./prods(k+1,:);
    Er(k,:)=prods(k+1,:)./prods(1,:);
end
[Erm0, J]=sort(Er(1,:), 'descend');  % Con la 1 norma
for i=1:nmet-1
	for j=1:np
        Erm(i,j)=Er(i,J(j));
    end
end
%fprintf('Resultados del ratio de productos\n');
%for i=1:nmet-1
%    fprintf('Ratio %d: de %.2f a %.2f\n',i,min(Erm(i,:)),max(Erm(i,:)));
%end

sim{1}='+k'; sim{2}='xk'; sim{3}='sk'; sim{4}='vk'; sim{5}='*k'; sim{6}='ok'; sim{7}='dk'; sim{8}='pk'; sim{9}='hk';
t=['semilogy(1:np,Erm(',int2str(1),',:),''',sim{1},''''];
for i=2:nmet-1
    t=[t,',1:np,Erm(',int2str(i),',:),''',sim{i},''''];
end
t=[t,')'];
eval(t);

t='legend(';
for j=2:nmet
    %G=['P(',F{1},')/P(',F{j},')'];
    G=['P(',F{j},')/P(',F{1},')'];
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
ylabel('Matrix product ratio')

% Comentado para cambiar el nombre al fichero de la gráfica
%aux=['Figures\Fig_',test,'_matrix_product_ratio'];
aux=['Figures\matrix_product_ratio_',test];
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
fprintf('\nNúmero total de productos:\n');
for j=1:nmet
    fprintf('\t%15s: %g\n',f{j},sum(prods(j,:)));
end
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nValores de m y s:\n');
for j=1:nmet
    fprintf('\t%15s: min(m)=%5.2f, max(m)=%5.2f, media(m)=%5.2f. ',f{j},min(M(j,:)),max(M(j,:)),sum(M(j,:))/np);
    fprintf('min(s)=%5.2f, max(s)=%5.2f, media(s)=%5.2f\n',min(S(j,:)),max(S(j,:)),sum(S(j,:)/np));
end
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
fprintf('%.1f*E(%s)<=E(%s):%.2f\n',alfa(lalfa),f1,f2,I(lalfa));
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
fprintf('E(%s)=E(%s) en %.2f%% ocasiones\n',f1,f2,igual/np*100);
fprintf('\n');
end
