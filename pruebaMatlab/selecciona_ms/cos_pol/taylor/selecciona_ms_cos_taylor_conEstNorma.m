function [m,s,pB,nProd]=selecciona_ms_cos_taylor_conEstNorma(plataforma,A)
% [m,s,pB,nProd]=selecciona_ms_cos_taylor_conEstNorma(plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz, con estimaciones de la norma de las potencias 
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
%               pB{i} contiene B^i, para i=1,2,3,...,q, siendo q<=4.
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

% m: 1 2 4  6    9   12   16
% q: 1 2 2  2,3  3   3,4  4
% Theta from forward and backward error bounds
Theta=[5.161913593731081e-08      % m=1  forward bound
       4.307691256676447e-05      % m=2  forward bound
       1.319680929892753e-02      % m=4  forward bound
       1.895232414039165e-01      % m=6  forward bound
       1.7985058769167590         % m=9  backward bound
       6.752349007371135          % m=12 backward bound
       9.971046342716772          % m=16 backward bound
      10.177842844012551];        % m=20 backward bound

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
        pB{1}=call_gpu('power');         
        d(1)=call_gpu('norm1',1);
end
nProd=1;
q=1;
b(1)=d(1);
if d(1)<=Theta(1)
    m=1;
    return
end

% m = 2
switch plataforma
    case 'sinGPUs'
        pB{2}=pB{1}*pB{1};
        d(2)=norm(pB{2},1);
    case 'conGPUs'
        pB{2}=call_gpu('power');
        d(2)=call_gpu('norm1',2);
end
nProd=nProd+1;
q=2;
b(2)=d(2)^(1/2);
if d(2)==0
    m=1;
    q=1;
    % Liberamos la potencia más alta calculada, al ser innecesaria
    switch plataforma
        case 'sinGPUs'
            pB(2)=[];
        case 'conGPUs'
            pB(2)=[];
            call_gpu('free',1);
    end   
    return
end %Nilpotent matrix

beta_min = beta_cos_taylor_sinEstNorma(b,d,2,q);
if beta_min<=Theta(2)
    m=2; % q=2
    return
end

% m = 4
beta_min = min(beta_min,beta_cos_taylor_sinEstNorma(b,d,4,q));
if beta_min<=Theta(3)
    d(3)=norm1pp(pB{2},1,pB{1});b(3)=d(3)^(1/3); %Test if previous m = 2 is valid
    if b(3)<=Theta(2)
        d(4)=min(d(2)^2,d(3)*d(1));
        b(4)=d(4)^(1/4);
        if b(4)<=Theta(2)
            m=2; % q=2
            return
        end
        d(4)=norm1p(pB{2},2);
        b(4)=d(4)^(1/4);
        if b(4)<=Theta(2)
            m=2; % q=2
            return
        end
    end
    m=4; % q=2
    return
end

% m = 6
d(6)=min(d(3)^2,d(2)^3);
beta_min = min(beta_min,beta_cos_taylor_sinEstNorma(b,d,6,q));
if beta_min<=Theta(4)
    d(5)=norm1pp(pB{2},2,pB{1});
    b(5)=d(5)^(1/5); %Test if previous m = 4 is valid
    if b(5)<=Theta(3)
        d(6)=min(d(5)*d(1),d(2)^3);
        b(6)=d(6)^(1/6);  
        if b(6)<=Theta(3)
            m=4; % q=2
            return
        end %Considering only two terms
        d(6)=norm1p(pB{2},3);
        b(6)=d(6)^(1/6);
        if b(6)<=Theta(3)
            m=4; % q=2
            return
        end
    end
    m=6; % q=2
    return
end

switch plataforma
    case 'sinGPUs'
        pB{3}=pB{2}*pB{1};
        d(3)=norm(pB{3},1);
    case 'conGPUs'
        pB{3}=call_gpu('power');
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
            pB(3)=[];
            call_gpu('free',1);
    end    
    return
end %Nilpotent matrix

d(6)=min(d(3)^2,d(2)^3);
beta_min = min(beta_min,beta_cos_taylor_sinEstNorma(b,d,6,q));
if beta_min<=Theta(4)
    m=6; % q=3
    return
end

% m = 9
d(9)=d(6)*d(3);
b(9)=d(9)^(1/9);
d(10)=min(d(9)*d(1),d(6)*d(2)^2);
b(10)=d(10)^(1/10);
beta_min9=min(beta_min,max(b(9:10)));      %Considering only two terms
if beta_min9<=Theta(5) 
    d(7)=norm1pp(pB{3},2,pB{1});
    b(7)=d(7)^(1/7);  % Test if previous m = 6 is valid
    if b(7)<=Theta(4)
        d(8)=min(d(7)*d(1),d(6)*d(2));
        b(8)=d(8)^(1/8);
        if b(8)<=Theta(4)
            m=6; % q=3
            return 
        end
        d(8)=min(d(8),norm1p(pB{2},4));
        b(8)=d(8)^(1/8);
        if b(8)<=Theta(4)
            m=6; % q=3
            return
        end
    end
    m=9; % q=3
    return
end

% Orders for scaling
d(12)=norm1p(pB{3},4);
b(12)=d(12)^(1/12);
d(13)=norm1pp(pB{3},4,pB{1});
b(13)=d(13)^(1/13);
beta_min12=max(b(12:13)); % Considering only two terms
s9aux=max(0,ceil(log2(beta_min12/Theta(5))/2));
s12=max(0,ceil(log2(beta_min12/Theta(6))/2));
if s9aux<=s12 % No scaling (s=0) included
    s9=max(0,ceil(log2(beta_min9/Theta(5))/2));
    if s9<=s12
        m=9; % q=3
        s=s9; 
        return
    end
    d(9)=norm1p(pB{3},3);
    b(9)=d(9)^(1/9);
    d(10)=min(d(9)*b(1),d(6)*d(2)^2);
    b(10)=d(10)^(1/10);
    beta_min9=min(beta_min9,max(b(9:10)));
    s9=max(0,ceil(log2(beta_min9/Theta(5))/2));
    if s9<=s12
        m=9; % q=3
        s=s9; 
        return
    end
    d(10)=norm1pp(pB{3},3,pB{1});
    b(10)=d(10)^(1/10);
    beta_min9=min(beta_min9,max(b(9:10)));
    s9=max(0,ceil(log2(beta_min9/Theta(5))/2));
    if s9<=s12
        m=9; % q=3
        s=s9; 
        return
    end
end

switch plataforma
    case 'sinGPUs'
        pB{4}=pB{3}*pB{1};
        d(4)=norm(pB{4},1);
    case 'conGPUs'
        pB{4}=call_gpu('power');
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
            pB(4)=[];
            call_gpu('free',1);
    end      
    return
end % Nilpotent matrix

d(6)=min(d(6),d(4)*d(2));
d(9)=min([d(9),d(4)^2*d(1),d(4)*d(3)*d(2)]); % d(9)  may have been estimated
d(10)=min([d(10),d(4)^2*d(2),d(6)*d(4)]);    % d(10) may have been estimated
d(16)=min([d(12)*d(4),d(13)*d(3),d(9)*d(4)*d(3),d(10)*d(6)]);
b(16)=d(16)^(1/16);
d(17)=min([d(12)*min(d(3)*d(2),d(4)*d(1)),d(13)*d(4),d(9)*min(d(4)^2,d(6)*d(2)),d(10)*d(4)*d(3)]);
b(17)=d(17)^(1/17);
beta_min16 = min(beta_min12,max(b(16:17)));
s12=max(0,ceil(log2(beta_min12/Theta(6))/2));
if s12==0
    m=12; % q=4
    s=0;
    return
end

s16=max(0,ceil(log2(beta_min16/Theta(7))/2));
if s12<=s16
    d(16)=min(d(16),norm1p(pB{4},4));
    b(16)=d(16)^(1/16);
    s16aux=max(0,ceil(log2(b(16)/Theta(7))/2)); % Temptative scaling for m=16
    if s16aux<s12    
        d(17)=min(d(17),d(16)*b(1));
        b(17)=d(17)^(1/17);
        beta_min16=min(beta_min16,max(b(16:17)));
        s16=max(0,ceil(log2(beta_min16/Theta(7))/2));
        if s16<s12
            m=16; % q=4
            s=s16; 
            return
        end  % If m=16 has a lower scaling s16 and the same or less cost than m=12, m=16 is preferred
        d(17)=norm1pp(pB{4},4,pB{1});
        b(17)=d(17)^(1/17);
        beta_min16=min(beta_min16,max(b(16:17)));
        s16=max(0,ceil(log2(beta_min16/Theta(7))/2));
        if s16<s12
            m=16; % q=4
            s=s16; 
            return
        else
            m=12; % q=4
            s=s12; 
            return
        end % If m=16 has a lower scaling s16 and the same or less cost than m=12, m=16 is preferred
    else
        m=12; % q=4
        s=s12; 
        return
    end % If m=12 has the same or less cost than m=16, m=12 is preferred
else
    m=16; % q=4
    s=s16;
    return
end
error('selecciona_ms_cos_taylor_conEstNorma:NoParam','Cannot find valid parameters, check matrix')
end
