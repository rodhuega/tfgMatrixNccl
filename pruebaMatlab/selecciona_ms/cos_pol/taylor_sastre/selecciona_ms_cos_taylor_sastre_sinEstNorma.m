function [m,s,pB,nProd]=selecciona_ms_cos_taylor_sastre_sinEstNorma(plataforma,A)
% [m,s,pA,nProd]=selecciona_ms_cos_taylor_sastre_sinEstNorma(plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz, sin estimaciones de la norma de las potencias 
% matriciales, para aplicar la función coseno sobre A mediante Taylor.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
%
% Datos de salida:
% - m:          Orden de la aproximación polinómica a f(A). El grado del 
%               polinomio de aproximación a f(A) será 2*m.
% - s:          Valor del escalado de la matriz.
% - pB:         Vector de celdas con las potencias de B=A^2, de modo que
%               pB{i} contiene B^i, para i=1,2,3,...,q, siendo q<=3.
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

% m: 1 2 4 8 12 15
% q: 1 2 2 2  3  3

% Theta from forward and backward error bounds
Theta=[5.161913593731081e-08      %m=1  forward bound
       4.307691256676447e-05      %m=2  forward bound
       1.319680929892753e-02      %m=4  forward bound
       0.93933580333673539   %m=8  relative backward bound
       6.7523490074059511    %m=12 relative backward bound
       16.451238315562541];  %m=15 absolute forward bound Backward:9.9323454917022698 (1500 term, 150 vpa)


d=zeros(1,17); %||B^k||, B=A^2
b=zeros(1,17); %||B^k||^(1/k), B=A^2
%Initial scaling parameter s = 0 
s = 0;

% m = 1
switch plataforma
    case 'sinGPUs'
        pB{1}=A*A;
        d(1)=norm(pB{1},1);
     case 'conGPUs'
        pB=[];
        call_gpu('power'); 
        d(1)=call_gpu('norm1',1);
end
nProd=1;
q=1;
b(1)=d(1);
if b(1)<=Theta(1)
    m=1; 
    return
end

% m = 2
switch plataforma
    case 'sinGPUs'
        pB{2}=pB{1}*pB{1};
        d(2)=norm(pB{2},1);
    case 'conGPUs'
        call_gpu('power');
        d(2)=call_gpu('norm1',2);
end
nProd=nProd+1;
q=2;
b(2)=d(2)^(1/2);
if b(2)==0
    m=1;
    q=1;
    % Liberamos la potencia más alta calculada, al ser innecesaria
    switch plataforma
        case 'sinGPUs'
            pB(2)=[];
        case 'conGPUs'
            call_gpu('free',1);
    end      
    return
end %Nilpotent matrix

beta_min = beta_cos_taylor_sastre_sinEstNorma(b,d,2);
if beta_min<=Theta(2)
    m=2; % q=2
    return
end

% m = 4
beta_min = min(beta_min,beta_cos_taylor_sastre_sinEstNorma(b,d,4));
if beta_min<=Theta(3)
    m=4; % q=2
    return
end

% m = 8
beta_min = min(beta_min,beta_cos_taylor_sastre_sinEstNorma(b,d,8));
if beta_min<=Theta(4)
    m=8; % q=2
    return
end

% m = 12
switch plataforma
    case 'sinGPUs'
        pB{3}=pB{2}*pB{1};
        d(3)=norm(pB{3},1);
    case 'conGPUs'
        call_gpu('power');
        d(3)=call_gpu('norm1',3);
end
nProd=nProd+1;
q=3;
b(3)=d(3)^(1/3);
if b(3)==0
    m=2; 
    q=2;
    % Liberamos la potencia más alta calculada, al ser innecesaria
    switch plataforma
        case 'sinGPUs'
            pB(3)=[];
        case 'conGPUs'
            call_gpu('free',1);
    end
    return
end %Nilpotent matrix

beta_min = min(beta_min,beta_cos_taylor_sastre_sinEstNorma(b,d,12));
if beta_min<=Theta(5)
    m=12; % q=3
    return
end
s12 = ceil(log2(beta_min/Theta(5))/2);
beta_min = min(beta_min,beta_cos_taylor_sastre_sinEstNorma(b,d,15));
s15 = max(0,ceil(log2(beta_min/Theta(6))/2)); %Scaling s=0 included

%m=12 only used for scaling if cost is lower than cost with m=15
if s12<=s15
    m=12; % q=3
    s=s12;
    return
else
    m=15; % q=3
    s=s15; 
    return
end
error('selecciona_ms_cos_taylor_sastre_sinEstNorma:NoParam','Cannot find valid parameters, check matrix')
end