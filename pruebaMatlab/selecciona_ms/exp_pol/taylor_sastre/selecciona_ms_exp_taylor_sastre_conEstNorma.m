function [m,s,pA,nProd]=selecciona_ms_exp_taylor_sastre_conEstNorma(plataforma,A,mmax)
% [m,s,pA,nProd]=selecciona_ms_exp_taylor_sastre_conEstNorma(plataforma,A,mmax)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz, con estimaciones de la norma de las potencias 
% matriciales, para aplicar la función exponencial sobre A mediante Taylor.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
% - mmax:       Valor máximo del grado del polinomio de aproximación.
%               Valores posibles son 24 o 30.
%
% Datos de salida:
% - m:          Orden de la aproximación polinómica a f(A). Coincide con el 
%               grado del polinomio de aproximación a f(A).
% - s:          Valor del escalado de la matriz.
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i}
%               contiene A^i, para i=1,2,3,...,q, con q<=5.
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

[m,s,q,maxpow,pA,nProd]=selecciona_ms_exp_taylor_sastre_conEstNorma_sub(plataforma,A,mmax);

% Scaling

if maxpow<=3 && q>3
    switch plataforma
        case 'sinGPUs'
            pA{4}=pA{3}*pA{1};
            a4=norm(pA{4},1);    
        case 'conGPUs'
            pA{4}=call_gpu('power');
            a4=call_gpu('norm',4);
    end    
    nProd=nProd+1;
    if a4==0
        m=3; 
        q=3;
        s=0;
        % Liberamos la potencia más alta calculada, al ser innecesaria
        switch plataforma
            case 'sinGPUs'
                pA(4)=[];
            case 'conGPUs'
                pA(4)=[];
                call_gpu('free',1);
        end         
    end  %Nilpotent matrix 
end
if maxpow<=4 && q>4
    switch plataforma
        case 'sinGPUs'
            pA{5}=pA{4}*pA{1};
            a5=norm(pA{4},1);    
        case 'conGPUs'
            pA{5}=call_gpu('power');
            a5=call_gpu('norm',5);
    end      
    nProd=nProd+1;
    if a5==0
        m=4;
        q=4;
        s=0;
        % Liberamos la potencia más alta calculada, al ser innecesaria
        switch plataforma
            case 'sinGPUs'
                pA(5)=[];
            case 'conGPUs'
                pA(5)=[];
                call_gpu('free',1);
        end         
    end  %Nilpotent matrix    
end
end

function [m,s,q,maxpow,pA,nProd]=selecciona_ms_exp_taylor_sastre_conEstNorma_sub(plataforma,A,mmax)
% [m,s,pA,nProd]=selecciona_ms_exp_taylor_sastre_conEstNorma(plataforma,A,mmax)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz, con estimaciones de la norma de las potencias 
% matriciales, para aplicar la función exponencial sobre A mediante Taylor.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
% - mmax:       Valor máximo del grado del polinomio de aproximación.
%               Valores posibles son 24 o 30.
%
% Datos de salida:
% - m:          Orden de la aproximación polinómica a f(A). Coincide con el 
%               grado del polinomio de aproximación a f(A).
% - s:          Valor del escalado de la matriz.
% - q:          Máxima potencia matricial a ser usada en la aproximación
%               de Taylor(A,A^2,...,A^q).
% - maxpow:     Máxima potencia matricial computada en la aproximación
%               de Taylor(maxpow<=q).
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i}
%               contiene A^i, para i=1,2,3,...,q, con q<=5.
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

% Optimal orders m:            1 2 4 8 15 21 24 30
% Corresponding matrix powers: 1 2 2 2  2  3  4 5
thetam1=1.490116111983279e-8; %m=1  Forward bound
theta=[8.733457513635361e-006 %m=2  Forward bound
    1.678018844321752e-003    %m=4  Forward bound
    6.950240768069781e-02     %m=8  Forward bound
    6.925462617470704e-01     %m=15 Forward bound
    1.682715644786316         %m=21 Backward bound
    2.219048869365090         %m=24 Backward bound
    3.539666348743690];       %m=30 Backward bound

c=[4/3    %m=2   %c=abs(c_m(m+1)/c_m(m+2)) (Table 5 of [1])
   6/5    %m=4
   10/9   %m=8
   1.148757271434994 %m=15 especial calculation
   1.027657297529898 %m=21 especial calculation
   26/25   %m=24
   32/31]; %m=30

ucm2=[8.881784197001252e-016  %m=2 %ucm2=abs(u/c_m(m+2)) (Table 5 of [1])
    1.598721155460225e-014  %m=4
    4.476419235288631e-011  %m=8
    5.874311180519475e-03   %m=15
    2.935676824339517e+05   %m=21
    1.790973863109916e+09   %m=24
    9.423674633957229e+017];%m=30

mv = [2 4 8 15 21 24 30];  %Vector of optimal orders m.
% Usaremos 24 o 30
if mmax<mv(6)
   mmax=mv(1);
elseif mmax>mv(7)
    mmax=mv(2);
end

% Buscamos la posición (kmax) de mmax en el vector mv
i=1;
encontrado=0;
while i<=length(mv) && encontrado==0
    if mv(i)==mmax
        kmax=i;
        encontrado=1;
    else
        i=i+1;
    end
end

%Initial scaling parameter s = 0
s = 0;
nProd = 0;
%pA=cell(5,1);
pA{1}=A;
a=realmax*ones(1,26); %vector of ||A^k|| (if a(i)=realmax: not calculated)

% m = 1
switch plataforma
    case 'sinGPUs'
        a(1)=norm(pA{1},1);    
    case 'conGPUs'
        a(1)=call_gpu('norm',1);
end
if a(1)<=thetam1
    q=1;
    maxpow=1;
    m=1;
    return 
end

% m = 2
switch plataforma
    case 'sinGPUs'
        pA{2}=pA{1}*pA{1};
        a(2)=norm(pA{2},1);    
    case 'conGPUs'
        pA{2}=call_gpu('power');
        a(2)=call_gpu('norm',2);
end
nProd=1; 
q=2;
maxpow=2;
if a(2)==0
    m=1;
    q=1;
    switch plataforma
        case 'sinGPUs'
            pA(2)=[];
        case 'conGPUs'
            pA(2)=[];
            call_gpu('free',1);
    end       
    return
end %Nilpotent matrix

a(3)=a(1)*a(2);
a(4)=a(2)^2; 
b(1)=max(1,a(1))*ucm2(1);
if c(1)*a(3)+a(4)<=b(1)
    m=2; % q=2
    return
end

% m = 4
a(5)=a(4)*a(1); 
a(6)=a(4)*a(2);
b(2)=max(1,a(1))*ucm2(2);
if c(2)*a(5)+a(6)<=b(2)
    a(3)=norm1pp(pA{2},1,A); %Test if previous m = 2 is valid
    if c(1)*a(3)<=b(1)
        a(4)=min(a(4),a(3)*a(1));
        if c(1)*a(3)+a(4)<=b(1)
            m=2; 
            return
        end
        a(4)=norm1p(pA{2},2);
        if c(1)*a(3)+a(4)<=b(1)
            m=2;
            return
        end
    end
    m=4;
    return
end 
% m = 8
a(9)=a(5)*a(4);
a(10)=a(6)*a(4);
b(3)=max(1,a(1))*ucm2(3);
if c(3)*a(9)+a(10)<=b(3)
    a(5)=norm1pp(pA{2},2,A); %Test if previous m = 4 is valid
    if c(2)*a(5)<=b(2)
        a(6)=min(a(6),a(5)*a(1));
        if c(2)*a(5)+a(6)<=b(2)
            m=4;
            return,
        end
        a(6)=norm1p(pA{2},3);
        if c(2)*a(5)+a(6)<=b(2)
            m=4;
            return
        end
    end  
    m=8;
    return
end 

% m = 15 no estimation
a(16)=a(2)^8;
a(17)=a(16)*a(1);
b(4)=max(1,a(1))*ucm2(4);
if c(4)*a(16)+a(17)<=b(4)
    a(9)=norm1pp(pA{2},4,A); %Test if previous m = 8 is valid
    if c(3)*a(9)<=b(3)
        a(10)=min(a(10),a(9)*a(1));
        if c(3)*a(9)+a(10)<=b(3)
            m=8;
            return
        end
        a(10)=norm1p(pA{2},5);
        if c(3)*a(9)+a(10)<=b(3)
            m=8;
            return
        end
    end      
    m=15;
    return 
end 

% m = 15 with estimation
a(16)=norm1p(pA{2},8);
a(17)=a(16)*a(1);
if c(4)*a(16)<=b(4)
    a(9)=norm1pp(pA{2},4,A); %Test if previous m = 8 is valid
    if c(3)*a(9)<=b(3)
        a(10)=min(a(10),a(9)*a(1));
        if c(3)*a(9)+a(10)<=b(3)
            m=8;
            return
        end
        a(10)=norm1p(pA{2},5);
        if c(3)*a(9)+a(10)<=b(3)
            m=8;
            return
        end
    end
    if c(4)*a(16)+a(17)<=b(4)
        m=15;
        return
    end
    a(17) = norm1pp(pA{2},8,A);
    if c(4)*a(16)+a(17)<=b(4)
        m=15;
        return
    end
end

% m = 21
switch plataforma
    case 'sinGPUs'
        pA{3}=pA{2}*pA{1};
        a(3)=norm(pA{3},1);    
    case 'conGPUs'
        pA{3}=call_gpu('power');
        a(3)=call_gpu('norm',3);
end
nProd=2;
q=3;
maxpow=3;
if a(3)==0
    m=2; 
    q=2;
    switch plataforma
        case 'sinGPUs'
            pA(3)=[];
        case 'conGPUs'
            pA(3)=[];
            call_gpu('free',1);
    end     
    return
end %Nilpotent matrix

a(4)=min(a(4),a(3)*a(1));
a(5)=min(a(5),a(2)*a(3));
a(6)=min(a(6),a(3)^2);
a(7)=min(a(5)*a(2),a(6)*a(1));
a(22)=min([a(16)*a(6),a(17)*a(5),a(2)^11,a(3)^6*a(2)^2,a(3)^7*a(1)]); 
a(23)=min([a(16)*a(7),a(17)*a(6),a(2)^10*a(3),a(3)^7*a(2)]);
b(5)=max(1,a(1))*ucm2(5);
if c(5)*a(22)+a(23)<=b(5)
    m=21; 
    return
end

% m = 24
a(8)=a(6)*a(2);
a(9)=min(a(9),a(3)^3);
a(10)=min(a(10),a(9)*a(1));
a(25)=min([a(16)*a(9),a(17)*a(8),a(2)^11*a(3),a(3)^7*a(2)^2,a(3)^8*a(1)]);
a(26)=min([a(16)*a(10),a(17)*a(9),a(2)^13,a(3)^8*a(2)]);
b(6)=max(1,a(1))*ucm2(6);
if c(6)*a(25)+a(26)<=b(6)
    a(22)=norm1pp(pA{3},7,A); %Test if previous m = 21 is valid
    if c(5)*a(22)<=b(5)
        a(23)=min(a(23),a(22)*a(1));
        if c(5)*a(22)+a(23)<=b(5)
            m=21;
            return
        end
        a(23)=norm1pp(pA{3},7,pA{2});
        if c(5)*a(22)+a(23)<=b(5)
            m=21;
            return
        end
    end
    m=24;
    q=4;
    return
end

%Orders for scaling m=21 and mmax=24
if kmax==6 
    a(25)=min(a(25),norm1pp(pA{3},8,A));
    a(26)=min(a(26),a(25)*a(1));
    a26estimated=false;
    if c(6)*a(25)<=b(6)
        a(22)=norm1pp(pA{3},7,A); %Test if previous m = 21 is valid
        if c(5)*a(22)<=b(5)
            a(23)=min(a(23),a(22)*a(1));
            if c(5)*a(22)+a(23)<=b(5)
                m=21;
                return
            end
            a(23)=norm1pp(pA{3},7,pA{2});
            if c(5)*a(22)+a(23)<=b(5)
                m=21;
                return
            end
        end
        if c(6)*a(25)+a(26)<=b(6)
            m=24;
            q=4;
            return
        end
        a(26)=min(a(26),norm1pp(pA{3},8,pA{2}));
        a26estimated=true;
        if c(6)*a(25)+a(26)<=b(6)
            m=24;
            q=4; 
            return
        end
    end
    if ~a26estimated
        a(26)=min(a(26),norm1pp(pA{3},8,pA{2}));
    end    
    % Compute alpha_min for m=24 inm=6
    alpha_min = max(a(25)^(1/25), a(26)^(1/26));
    [t, s] = log2(alpha_min/theta(6));
    s = s - (t == 0.5); % adjust s if alpha_min/theta(kmax) is a power of 2.
    % Test if s can be reduced
    if s>0
        sred = s-1;
        b(6) = max(1,a(1)/2^sred)*ucm2(6);
        if c(6)*a(25)/2^(25*sred)+a(26)/2^(26*sred)<=b(6)
            % s can be reduced
            s = sred;
        end
    end
    % Test if the scaled matrix allows using m=21 inm=5
    a(22)=norm1pp(pA{3},7,A);
    a(23)=min(a(23),a(22)*a(1));
    b(5) = max(1,a(1)/2^s)*ucm2(5);
    if c(5)*a(22)/2^(22*s)<=b(5)
        if c(5)*a(22)/2^(22*s)+a(23)/2^(23*s)<=b(5)
            m=21;
            return
        end  % The scaled matrix allows using 21
        a(23)=norm1pp(pA{3},7,pA{2});
        if c(5)*a(22)/2^(22*s)+a(23)/2^(23*s)<=b(5)
            m=21;
            return
        end  % The scaled matrix allows using 21
    end
    m=24;
    q=4;
    return
%Orders for scaling m=24 and mmax=30
else
    % m = 24
    switch plataforma
        case 'sinGPUs'
            pA{4}=pA{3}*pA{1};
            a(4)=norm(pA{4},1);    
        case 'conGPUs'
            pA{4}=call_gpu('power');
            a(4)=call_gpu('norm',4);
    end    
    nProd=3; 
    q=4; 
    maxpow=4;
    if a(4)==0
        m=3;
        q=3;
        switch plataforma
            case 'sinGPUs'
                pA(4)=[];
            case 'conGPUs'
                pA(4)=[];
                call_gpu('free',1);
        end         
        return
    end %Nilpotent matrix
    
    a(25)=min([a(25),a(16)*a(4)^2*a(1),a(17)*a(4)^2,a(3)^7*a(4)]);
    a(26)=min([a(26),a(16)*a(4)^2*a(2),a(17)*a(4)^2*a(1),a(3)^7*a(4)*a(1)]);
    b(6)=max(1,a(1))*ucm2(6);
    if c(6)*a(25)+a(26)<=b(6)
        m=24;
        return
    end
    % m = 30
    a(14)=min(a(4)^3,a(3)^4)*a(2);
    a(15)=min(a(4)^3*a(3),a(3)^5);
    a(31)=min([a(16)*a(15),a(17)*a(14),a(3)^9*a(4)]);
    a(32)=min([a(16)^2,a(17)*a(15),a(3)^10*a(2),a(31)*a(1)]);
    b(7)=max(1,a(1))*ucm2(7);
    if c(7)*a(31)+a(32)<=b(7)
        a(25)=norm1pp(pA{4},6,A); %Test if previous m = 24 is valid
        if c(6)*a(25)<=b(6)
            a(26)=min(a(26),a(25)*a(1));
            if c(6)*a(25)+a(26)<=b(6)
                m=24;
                return
            end
            a(26)=norm1pp(pA{4},6,pA{2});
            if c(6)*a(25)+a(26)<=b(6)
                m=24;
                return
            end
        end
        m=30;
        q=5;
        return
    end
    a(31)=min(a(31),norm1pp(pA{4},7,pA{3}));
    a(32)=min(a(32),a(31)*a(1));
    a32estimated=false;
    if c(7)*a(31)<=b(7)
        a(25)=norm1pp(pA{4},6,A); %Test if previous m = 24 is valid
        if c(6)*a(25)<=b(6)
            a(26)=min(a(26),a(25)*a(1));
            if c(6)*a(25)+a(26)<=b(6)
                m=24; 
                return
            end
            a(26)=norm1pp(pA{4},6,pA{2});
            if c(6)*a(25)+a(26)<=b(6)
                m=24; 
                return
            end
        end
        if c(7)*a(31)+a(32)<=b(7)
            m=30;
            q=5; 
            return 
        end
        a(32)=min(a(32),norm1p(pA{4},8));
        a32estimated=true;
        if c(7)*a(31)+a(32)<=b(7)
            m=30; 
            q=5; 
            return,
        end
    end
    if ~a32estimated
        a(32)=min(a(32),norm1p(pA{4},8));
    end
    % Compute alpha_min for m=30 inm=7
    alpha_min = max(a(31)^(1/31), a(32)^(1/32));
    [t, s] = log2(alpha_min/theta(7));
    s = s - (t == 0.5); % adjust s if normA/theta(kmax) is a power of 2.
    % Test if s can be reduced
    if s>0
        sred = s-1;
        b(7) = max(1,a(1)/2^sred)*ucm2(7);
        if c(7)*a(31)/2^(31*sred)+a(32)/2^(32*sred)<=b(7)
            % s can be reduced
            s = sred;
        end
    end
    % Test if the scaled matrix allows using m=24 inm=6
    a(25)=norm1pp(pA{4},6,A);
    a(26)=min(a(26),a(25)*a(1));
    b(6) = max(1,a(1)/2^s)*ucm2(6);
    if c(6)*a(25)/2^(25*s)<=b(6)
        if c(6)*a(25)/2^(25*s)+a(26)/2^(26*s)<=b(6)
            m=24;
            return
        end  % The scaled matrix allows using 24
        a(26)=norm1pp(pA{4},6,pA{2});
        if c(6)*a(25)/2^(25*s)+a(26)/2^(26*s)<=b(6)
            m=24;
            return
        end  % The scaled matrix allows using 24
    end
    switch plataforma
        case 'sinGPUs'
            pA{5}=pA{4}*pA{1};
            a(5)=norm(pA{5},1);    
        case 'conGPUs'
            pA{5}=call_gpu('power');
            a(5)=call_gpu('norm',5);
    end     
    nProd=4;
    q=5;
    maxpow=5;
    if a(5)==0
        m=4;
        q=4;
        s=0;
        switch plataforma
            case 'sinGPUs'
                pA(5)=[];
            case 'conGPUs'
                pA(5)=[];
                call_gpu('free',1);
        end         
        return
    end %Nilpotent matrix
    m=30;
    q=5;
    return
end
end


