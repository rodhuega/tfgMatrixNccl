function [m,s,pA,nProd]=selecciona_ms_exp_splines_sinEstNorma(plataforma,A)
% [m,s,pA,nProd]=selecciona_ms_exp_splines_sinEstNorma(plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz, sin estimaciones de la norma de las potencias 
% matriciales, para aplicar la función exponencial sobre A mediante splines.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
%
% Datos de salida:
% - m:          Orden de la aproximación polinómica a f(A). Coincide con el 
%               grado del polinomio de aproximación a f(A).
% - s:          Valor del escalado de la matriz.
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i} 
%               contiene A^i, para i=1,2,3,...,q, siendo q=ceil(sqrt(m)).
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

Theta=[2.675298260329713e-8 %m=2  forward
       3.397168839977002e-4 %m=4  forward
       9.065656407595296e-3 %m=6  forward
       8.957760203223343e-2 %m=9  forward
       2.996158913811581e-1 %m=12 forward
       7.802874256626574e-1 %m=16 forward
       1.415070447561532    %m=20 backward
       2.353642766989427    %m=25 backward
       3.411877172556770];  %m=30 backward
   
nProd=0;  
%Initial scaling parameter s = 0 
s = 0;
% m = 1
switch plataforma
    case 'sinGPUs'
        pA{1}=A;
        a1=norm(pA{1},1);
    case 'conGPUs'
        pA=[];
        a1=call_gpu('norm1',1);
end

switch plataforma
    case 'sinGPUs'
        pA{2}=pA{1}*pA{1};
        a2=norm(pA{2},1);
    case 'conGPUs'
        call_gpu('power'); 
        a2=call_gpu('norm1',2);
end
nProd=nProd+1;

% m = 2
alfa=a2^(1/2);
if alfa<=Theta(1) % m=2 forward
    m=2; % q=2;
    return
end

% m = 4
a4=a2^2;alfa=a4^(1/4);
if alfa<=Theta(2) % m=4 forward
    m=4; % q=2;
    return
end

% m = 6
switch plataforma
    case 'sinGPUs'
        pA{3}=pA{2}*pA{1};
        a3=norm(pA{3},1);
    case 'conGPUs'
        call_gpu('power'); 
        a3=call_gpu('norm1',3);
end
nProd=nProd+1;

a6=min([a3^2,a2^3]);
alfa=a6^(1/6);
if alfa<=Theta(3)% m=6 forward
    m=6; % q=3;
    return
end

% m = 9
a9=a3*a6;
alfa=a9^(1/9);
if alfa<=Theta(4 )% m=9 forward
    m=9; % q=3;
    return
end

% m = 12
switch plataforma
    case 'sinGPUs'
        pA{4}=pA{3}*pA{1};
        a4=norm(pA{4},1);
    case 'conGPUs'
        call_gpu('power'); 
        a4=call_gpu('norm1',4);
end
nProd=nProd+1;

a12=min([a3*a9,a4*a6*a2,a4^3]);
alfa=a12^(1/12);
if alfa<=Theta(5)% m=12 forward
    m=12; % q=4;
    return
end

% m = 16
a16=a4*a12;
alfa=a16^(1/16);
if alfa<=Theta(6)% m=16 forward
    m=16; % q=4;
    return
end

% m = 20
switch plataforma
    case 'sinGPUs'
        pA{5}=pA{4}*pA{1};
        a5=norm(pA{5},1);
    case 'conGPUs'
        call_gpu('power'); 
        a5=call_gpu('norm1',5);
end
nProd=nProd+1;

a20=min([a4*a16,a5*a12*a3,a5^2*a9*a1,a5^2*a4^2*a2,a5^4]);
alfa=a20^(1/20);
if alfa<=Theta(7)% m=20 backward
    m=20; % q=5;  
    return
end

% m = 25
a25=a5*a20;
alfa25=a25^(1/25);
if alfa25<=Theta(8)% m=25 backward
    m=25; % q=5;
    return
end

% m = 30
switch plataforma
    case 'sinGPUs'
        pA{6}=pA{5}*pA{1};
        a6=norm(pA{6},1);
    case 'conGPUs'
        call_gpu('power'); 
        a6=call_gpu('norm1',6);
end
nProd=nProd+1;

a30=min([a5*a25,a6^5,a6^3*a12,a6^3*a5^2*a2,a6^2*a16*a2,a6^2*a5^3*a3,a6^2*a5^2*a4^2,a6*a20*a4]);
alfa=a30^(1/30);
if alfa<=Theta(9)% m=30 backward
    m=30; % q=6       
    return
end
s=ceil(log2(alfa/Theta(9)));
s25=ceil(log2(alfa25/Theta(8)));
if s25+1>=s
    m=30; % q=6
else
    m=25; % q=5
	s=s25;
    % Liberamos la potencia más alta calculada, al ser innecesaria
    switch plataforma
        case 'sinGPUs'
            pA(6)=[];
        case 'conGPUs'
            call_gpu('free',1);
    end     
end
end
