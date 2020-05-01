function pr_met_expv(f,dat,tipog,pr)
% pr_met_expv(f,dat,tipog,pr)
% f: funciones a comparar en forma de array de celdas
% dat: vector con el 2º argumento de cada función (cosher(A,7)).
%      Si no tiene argumentos la función, su valor es -1.
% tipog: tipo de gráfico guardado (1:fig,2:jpg,3:eps)
% pr: tipos de matrices (1, 2, 3, 4 o 5)
% Ejemplo:
% f{1}='expmvtay1';f{2}='expmvtay_inef1';f{3}='expmvhigham';f{4}='expmvhigham_inef';dat=[-1 -1 -1 -1];pr_met_expv(f,dat,1,5)

switch pr
    case 1 % OLVIDARSE DE ESTE CASO
        %test='exp_diag_nd256_n128'; %Estas matrices tienen un número de condición muy elevado
        test='exp_diag_128'; % Matrices reales
        nmat=100;
        [T,E,cond,prods]=calcula_errores(test,f,dat,nmat);
	case 2
        %test='expv_jordan_128';
        test='expv_jordan_256';
        nmat=100;
        [T,E,cond,prods]=calcula_errores(test,f,dat,nmat);
	case 3
        %test='exp_jordan_hadamard_128_nd128_mul10_alfa3';
        test='exp_eigtool_met_eig_nd_256_n128';
        nmat=7; 
        [T,E,cond,prods]=calcula_errores(test,f,dat,nmat);
        test='toolbox_exp_n128_nd32';
        nmat=31;
        [T0,E0,cond0,prods0]=calcula_errores(test,f,dat,nmat);
        T=[T,T0];prods=[prods,prods0];cond=[cond,cond0];E=[E,E0];
        %MS=[MS;MS2];
	case 4
        test='exp_several';
        nmat=27;
        [T,E,cond,prods]=calcula_errores(test,f,dat,nmat);
     case 5
        test='expv_diag_complex_nd256_n128'; % MATRICES GENERADAS POR MÍ
        nmat=100;
        [T,E,cond,prods]=calcula_errores(test,f,dat,nmat);
end
make_graph(test,f,T,E,cond,prods,14,tipog,0,pr)
end

function [T,E,cond,Np]=calcula_errores(test,f,dat,nmat)
nf=length(f);T=zeros(nf,nmat);Np=zeros(nf,nmat);
k=0;
for i=1:nmat
    eval(['load ',test,'_',int2str(i)]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if dat(1)>=0
        tic
        [C,~,~,np] = feval(f{1},A,v,dat(1));
        t=toc;
    else
        tic
        [C,~,~,np] = feval(f{1},A,v);
        t=toc;
    end
    Er=norm(expAv-C)/norm(expAv);
	if norm(Er)>1%Esto lo hago para evitar casos de mucho error
        continue
    end
    k=k+1;
    cond(k)=condA;
    T(1,k)=t;
    E(1,k)=Er;
    Np(1,k)=np;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j=2:nf
        if dat(j)>=0
            tic
            [C,~,~,np] = feval(f{j},A,v,dat(j));
            t=toc;           
        else
            tic
            [C,~,~,np] = feval(f{j},A,v);
            t=toc;  
        end
        T(j,k)=t;
        E(j,k)=norm(expAv-C)/norm(expAv);
        Np(j,k)=np;
    end
end
end


function make_graph(test,f,T,E,cond,prods,fontsize,tipog,mod,pr)
try
    warning off
	mkdir Figures
catch
end
if mod==1
    s='Figures\Fig_';
else
    s='Figures\Figm_';
end
nmet=size(E,1);
np=size(E,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Normwise
close all
figure(1)
[condAm, J]=sort(cond, 'descend');  % Con la 1 norma
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
hold on;grid on
t='legend(''cond*u''';
t=[t,',','''',f{1},''''];
t=[t,',','''',f{2},''''];
t=[t,',','''',f{3},''''];
t=[t,',','''',f{4},''',''location''',',''northeast',''')'];
%t=[t,',','''','expm\_new'',','''location''',',''northeast',''')'];
%t=[t,',''location''',',''northeast',''')'];
eval(t);
hold on;grid on
set(gca,'FontSize',fontsize);
xlabel('Matrix')
ylabel('Er')
if pr==3
     axis([0 np+1 0 2]);
end
aux=['Figures\Fig_',test,'_normwise_',f{1},'_',f{2},'_',f{3}];
salva_grafica(tipog,gcf,aux);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Performance profile
close all
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
hold on;grid on
set(gca,'FontSize',fontsize);
t='legend(';
t=[t,'''',f{1},''''];
t=[t,',','''',f{2},''''];
t=[t,',','''',f{3},''''];
t=[t,',','''',f{4},''',''location''',',''northeast',''')'];
%t=[t,',','''','expm\_new'',','''location''',',''southeast',''')'];

eval(t);
xlabel('\it \alpha')
ylabel('p')
%aux=[s,test,'_nprofile_',f{1},'_',f{2},'_',f{3},'_expm_new'];
aux=[s,test,'_nprofile_',f{1},'_',f{2},'_',f{3},'_',f{4}];
salva_grafica(tipog,gcf,aux);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Errores relativos
close all
figure(3)
for k=1:nmet-1
    Er(k,:)=E(1,:)./E(k+1,:);
end
[Erm0, J]=sort(Er(nmet-1,:), 'descend');  % Con la 1 norma
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
t0=[t,')'];
eval(t0);
t='legend(';
G2=['E(',f{1},')/E(',f{2},')'];
G3=['E(',f{1},')/E(',f{3},')'];
%G4=['E(',f{1},')/E(','expm\_new',')'];
G4=['E(',f{1},')/E(',f{4},')'];
t=[t,'''',G2,''''];
t=[t,','];
t=[t,'''',G3,''''];
t=[t,','];
t=[t,'''',G4,''''];
t=[t,')'];
eval(t);
hold on;grid on
set(gca,'FontSize',fontsize);
xlabel('Matrix')
ylabel('Ratio relative errors')
axis([0 np+1 -1 10]);
%aux=[s,test,'_ratio_errors_',f{1},'_',f{2},'_',f{3},'_expm_new'];
aux=[s,test,'_ratio_errors_',f{1},'_',f{2},'_',f{3},'_',f{4}];
salva_grafica(tipog,gcf,aux);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Productos relativos
close all
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
t=['plot(1:np,Erm(',int2str(1),',:),''',sim{1},''''];
for i=2:nmet-1
    t=[t,',1:np,Erm(',int2str(i),',:),''',sim{i},''''];
end
grid
t=[t,')'];
eval(t);
hold on;grid on
t='legend(';
G2=['M(',f{1},')/M(',f{2},')'];
G3=['M(',f{1},')/M(',f{3},')'];
%G4=['M(',f{1},')/M(','expm\_new',')'];
G4=['M(',f{1},')/M(',f{4},')'];
t=[t,'''',G2,''''];
t=[t,','];
t=[t,'''',G3,''''];
t=[t,','];
t=[t,'''',G4,''''];
t=[t,')'];
eval(t);
set(gca,'FontSize',fontsize);
grid
xlabel('Matrix')
ylabel('Ratio matrix products')
%aux=[s,test,'_ratio_products_',f{1},'_',f{2},'_',f{3},'_expm_new'];
aux=[s,test,'_ratio_products_',f{1},'_',f{2},'_',f{3},'_',f{4}];
salva_grafica(tipog,gcf,aux);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Tiempos relativos
figure(4)
hold on
grid on
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
t=['plot(1:np,Erm(',int2str(1),',:),''',sim{1},''''];
for i=2:nmet-1
    t=[t,',1:np,Erm(',int2str(i),',:),''',sim{i},''''];
end
grid
t=[t,')'];
eval(t);
t='legend(';
G2=['M(',f{1},')/M(',f{2},')'];
G3=['M(',f{1},')/M(',f{3},')'];
%G4=['M(',f{1},')/M(','expm\_new',')'];
G4=['M(',f{1},')/M(',f{4},')'];
t=[t,'''',G2,''''];
t=[t,','];
t=[t,'''',G3,''''];
t=[t,','];
t=[t,'''',G4,''''];
t=[t,')'];
eval(t);
set(gca,'FontSize',fontsize);
grid
xlabel('Matrix')
ylabel('Ratio matrix products')
%aux=[s,test,'_ratio_products_',f{1},'_',f{2},'_',f{3},'_expm_new'];
aux=[s,test,'_ratio_products_',f{1},'_',f{2},'_',f{3},'_',f{4}];
salva_grafica(tipog,gcf,aux);
grid off
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nComparativa entre %s y %s para %d matrices del test %s:\n',f{1},f{2},np,test);
%compara(E(1:2,:),f1,f2);
fprintf('\n');
comparam(E(1:2,:),f{1},f{2});
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nComparativa entre %s y %s para %d matrices del test %s:\n',f{1},f{3},np,test);
%compara([E(1,:);E(3,:)],f1,f3);
fprintf('\n');
comparam([E(1,:);E(3,:)],f{1},f{3});
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nComparativa entre %s y %s para %d matrices del test %s:\n',f{1},f{4},np,test);
%compara([E(1,:);E(4,:)],f1,f4);
fprintf('\n');
comparam([E(1,:);E(4,:)],f{1},f{4});
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Tiempo total:\n');
fprintf('%10s: %f \t %10s: %f %10s: %f \t %10s: %f \n',f{1},sum(T(1,:)),f{2},sum(T(2,:)),f{3},sum(T(3,:)),f{4},sum(T(4,:)));
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Nº productos total:\n');
for k=1:nmet
    P(k)=sum(prods(k,:));
end
P=ceil(P);
fprintf('\t%10s: %d\t%10s: %d\t%10s: %d\t%10s: %d\n',f{1},P(1),f{2},P(2),f{3},P(3),f{4},P(4));

grid off
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
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function salva_grafica(tipo,gcf,aux)
switch tipo
    case 1
        saveas(gcf,aux, 'fig');
    case 2
        saveas(gcf,aux, 'jpg');
    case 3
        set(gcf, 'Color', [1,1,1]);
        saveas(gcf,aux, 'epsc');  %% Changed for epsc files
end
end