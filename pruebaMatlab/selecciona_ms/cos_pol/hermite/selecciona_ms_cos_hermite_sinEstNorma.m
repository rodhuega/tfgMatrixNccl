function [m,s,pB,nProd]=selecciona_ms_cos_hermite_sinEstNorma(plataforma,A)
% [m,s,pB,nProd]=selecciona_ms_cos_hermite_sinEstNorma(plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz, sin estimaciones de la norma de las potencias 
% matriciales, para aplicar la función coseno sobre A mediante Hermite.
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
%               pB{i} contiene B^i, para i=1,2,3,...,q, siendo q<=4.
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

% m: 1 2 4 6 9 12 16
% q: 1 2 2 3 3  4  4
% Theta from forward and backward error bounds

Theta=[ 
        3.7247156392713732e-05  % m=2
        0.011723485672548786    % m=4
        0.17002640749360201     % m=6
        1.623738709885105       % m=9
        6.16270324616079        % m=12
        20.113380730824375      % m=16
      ];
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
b(1)=d(1);

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

beta_min = beta_cos_hermite_sinEstNorma(b,d,2,q);
if beta_min<=Theta(1)
    m=2; % q=2
    return
end

% m = 4
beta_min = min(beta_min,beta_cos_hermite_sinEstNorma(b,d,4,q));
if beta_min<=Theta(2)
    m=4; % q=2
    return
end

% m = 6
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

d(6)=min(d(3)^2,d(2)^3);
beta_min = min(beta_min,beta_cos_hermite_sinEstNorma(b,d,6,q));
if beta_min<=Theta(3)
    m=6; % q=3 
    return
end

% m = 9
beta_min9 = min(beta_min,beta_cos_hermite_sinEstNorma(b,d,9,q));
if beta_min9<=Theta(4)
    m=9; % q=3
    return
end

% m = 12
beta_min12 = min(beta_min9,beta_cos_hermite_sinEstNorma(b,d,12,q));
if beta_min12<=Theta(5)
    m=12;
    % AÑADIDO. FALTA LA POTENCIA CUARTA Y QUE q SEA IGUAL A 4, SI m VALE 12
    q=4;
    switch plataforma
        case 'sinGPUs'
            pB{4}=pB{3}*pB{1};
        case 'conGPUs'
            call_gpu('power');
    end
    nProd=nProd+1;
    % FIN AÑADIDO    
    return
end

%m=9 only used for scaling if cost is lower than cost with m=12,16
s9 = ceil(log2(beta_min9/Theta(4))/2); %Scaling s=0 not included
s12 = ceil(log2(beta_min12/Theta(5))/2);
if s9<=s12
    m=9; % q=3
    s=s9;
    return
end

% m = 12
switch plataforma
    case 'sinGPUs'
        pB{4}=pB{3}*pB{1};
        d(4)=norm(pB{4},1);
    case 'conGPUs'
        call_gpu('power');
        d(4)=call_gpu('norm1',4);
end
nProd=nProd+1;
q=4;
b(4)=d(4)^(1/4);
if b(4)==0
    m=3; % ¿ESTOS VALORES DE m Y q SON CORRECTOS?
    q=3; 
    % Liberamos la potencia más alta calculada, al ser innecesaria
    switch plataforma
        case 'sinGPUs'
            pB(4)=[];
        case 'conGPUs'
            call_gpu('free',1);
    end    
    return 
end %Nilpotent matrix

d(6)=min(d(6),d(4)*d(2));
beta_min12 = min(beta_min12,beta_cos_hermite_sinEstNorma(b,d,12,q)); %We have new information with pB{4}=B^4
if beta_min12<=Theta(5)
    m=12; % q=4
    s=0; 
    return
end

%m=12 only used for scaling if cost is lower than cost with m=16
s12 = ceil(log2(beta_min12/Theta(5))/2);
beta_min16 = min(beta_min12,beta_cos_hermite_sinEstNorma(b,d,16,q));
s16 = max(0,ceil(log2(beta_min16/Theta(6))/2)); %Scaling s=0 included
if s12<=s16
    m=12; % q=4
    s=s12;
    return 
else
    m=16; % q=4
    s=s16;
end 
end

