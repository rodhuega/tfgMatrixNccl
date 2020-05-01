function [m,s,pA,nProd]=selecciona_ms_exp_splines_conEstNorma(plataforma,A)
% [m,s,pA,nProd]=selecciona_ms_exp_splines_conEstNorma(plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz, con estimaciones de la norma de las potencias 
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

% Theta from forward and backward error bounds from [1] Table 2 
Theta=[2.675298260329713e-8 % m=2  forward
       3.397168839977002e-4 % m=4  forward
       9.065656407595296e-3 % m=6  forward
       8.957760203223343e-2 % m=9  forward
       2.996158913811581e-1 % m=12 forward
       7.802874256626574e-1 % m=16 forward
       1.415070447561532    % m=20 backward
       2.353642766989427    % m=25 backward
       3.411877172556770];  % m=30 backward
   
nProd=0;   
s = 0;
pA{1}=A;

switch plataforma
    case 'sinGPUs'
        pA{2}=pA{1}*A;
        a(2)=norm(pA{2},1);
    case 'conGPUs'
        pA{2}=call_gpu('power');
        a(2)=call_gpu('norm1',2);
end
nProd=nProd+1;

alfa=a(2)^(1/2);
if alfa<=Theta(1) % m=2 forward
    m=2; % q=2;
    return
end

alfa=norm1p(pA{2},2)^(1/4); % m=4 forward
if alfa<=Theta(2)
    m=4; % q=2;  
    return
end

% m = 6
switch plataforma
    case 'sinGPUs'
        pA{3}=pA{2}*A;
        %a(3)=norm(pA{3},1);
    case 'conGPUs'
        pA{3}=call_gpu('power');
        %a(3)=call_gpu('norm1',3);
end
nProd=nProd+1;

alfa=norm1p(pA{3},2)^(1/6); % m=6 forward
if alfa<=Theta(3)
    m=6; % q=3;   
    return
end

% m = 9
alfa=norm1p(pA{3},3)^(1/9); % m=9  forward
if alfa<=Theta(4)
    m=9; % q=3;  
    return
end

% m = 12
switch plataforma
    case 'sinGPUs'
        pA{4}=pA{3}*A;
        %a(4)=norm(pA{4},1);
    case 'conGPUs'
        pA{4}=call_gpu('power');
        %a(4)=call_gpu('norm1',4);
end
nProd=nProd+1;

alfa=norm1p(pA{4},3)^(1/12); % m=12 forward
if alfa<=Theta(5)
    m=12; % q=4;
    return
end

% m = 16
alfa=norm1p(pA{4},4)^(1/16); % m=16 forward
if alfa<=Theta(6)
    m=16; % q=4;
    return
end

% m = 20
switch plataforma
    case 'sinGPUs'
        pA{5}=pA{4}*A;
        %a(5)=norm(pA{5},1);
    case 'conGPUs'
        pA{5}=call_gpu('power');
        %a(5)=call_gpu('norm1',5);
end
nProd=nProd+1;
alfa=norm1p(pA{5},4)^(1/20); % m=20 backward
if alfa<=Theta(7)
    m=20;    % q=5;
    return
end

alfa25=norm1p(pA{5},5)^(1/25); % m=25 backward
if alfa25<=Theta(8)
    m=25;   % q=5;
    return
end

switch plataforma
    case 'sinGPUs'
        pA{6}=pA{5}*A;
        %a(6)=norm(pA{6},1);
    case 'conGPUs'
        pA{6}=call_gpu('power');
        %a(6)=call_gpu('norm1',6);
end
nProd=nProd+1;

alfa=norm1p(pA{6},5)^(1/30); % m=30 backward
if alfa<=Theta(9)
    m=30; % q=6;
    return
end
s= ceil(log2(alfa/Theta(9)));
s25=ceil(log2(alfa25/Theta(8)));
if s25+1>=s
    m=30; % q=6;
else
    m=25; % q=5;   
	s=s25;
    % Liberamos la potencia más alta calculada, al ser innecesaria
    switch plataforma
        case 'sinGPUs'
            pA(6)=[];
        case 'conGPUs'
            pA(6)=[];
            call_gpu('free',1);
    end    
end

end



