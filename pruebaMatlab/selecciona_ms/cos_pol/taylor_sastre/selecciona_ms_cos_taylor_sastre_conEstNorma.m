function [m,s,pB,nProd]=selecciona_ms_cos_taylor_sastre_conEstNorma(plataforma,A)
% [m,s,pA,nProd]=selecciona_ms_cos_taylor_sastre_conEstNorma(plataforma,A)
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


d=realmax*ones(1,17); %||B^k||, B=A^2
b=realmax*ones(1,17); %||B^k||^(1/k), B=A^2

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
    m=1; % q=1 
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

beta_min = beta_cos_taylor_sastre_sinEstNorma(b,d,2);
if beta_min<=Theta(2)
    m=2; % q=2
    return
end

% m = 4
beta_min = min(beta_min,beta_cos_taylor_sastre_sinEstNorma(b,d,4));
if beta_min<=Theta(3)
    d(3)=norm1pp(pB{2},1,pB{1});
    b(3)=d(3)^(1/3); %Test if previous m = 2 is valid
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

% m = 8
beta_min = min(beta_min,beta_cos_taylor_sastre_sinEstNorma(b,d,8));
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
    m=8; % q=2
    return
end
d(8)=norm1p(pB{2},4);
b(8)=d(8)^(1/8);
d(9)=d(8)*d(1);
b(9)=d(9)^(1/9);
if b(8)<=Theta(4)
    d(5)=norm1pp(pB{2},2,pB{1});
    b(5)=d(5)^(1/5); %Test if previous m = 4 is valid
    if b(5)<=Theta(3)
        d(6)=min(d(5)*d(1),d(2)^3);
        b(6)=d(6)^(1/6);
        if b(6)<=Theta(3)
            m=4; % q=2
            return
        end %Considering only two terms
        d(6)=norm1p(pB{2},3);b(6)=d(6)^(1/6);
        if b(6)<=Theta(3)
            m=4; % q=2
            return
        end
    end
    if b(9)<=Theta(4)
        m=8; % q=2
        return
    end %Considering only two terms
    d(9)=norm1pp(pB{2},4,pB{1});
    b(9)=d(9)^(1/9);
    if b(9)<=Theta(4)
        m=8; % q=2
        return
    end
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

d(9)=min([d(9),d(3)^3,d(2)^3*d(3)]);
d(4)=min(d(2)^2,d(3)*d(1));
d(12)=min(d(8)*d(4),d(9)*d(3));
b(12)=d(12)^(1/12);
d(13)=min(d(8)*d(2)*d(3),d(9)*d(4));
b(13)=d(13)^(1/13);
beta_min=max(b(12),b(13));
if beta_min<=Theta(5)
    m=12; % q=3 
    return
end

%Orders for scaling
d(12)=norm1p(pB{3},4);
b(12)=d(12)^(1/12);
s12aux1=max(0,ceil(log2(b(12)/Theta(5))/2)); %Tempative scaling for m = 12
d(13)=min(d(13),d(12)*d(1));
b(12)=d(12)^(1/12);
s12aux2=max(0,ceil(log2(b(13)/Theta(5))/2)); %Tempative scaling for m = 12
if s12aux2>s12aux1
    d(13)=norm1pp(pB{3},4,pB{1});
    b(13)=d(13)^(1/13); %Only computed if it may reduce the scaling
    s12aux2=max(0,ceil(log2(b(13)/Theta(5))/2)); %Tempative scaling for m = 12
end
s12=max(s12aux1,s12aux2);

%beta_min12=max(b(12:13)); %Considering only two terms
if s12==0
    m=12; % q=3
    return
end

%s12=ceil(log2(beta_min12/Theta(5))/2);
d(16)=min([d(12)*d(4),d(13)*d(3),d(8)^2]);b(16)=d(16)^(1/16);
d(17)=min([d(12)*d(3)*d(2),d(13)*d(4),d(16)*d(1)]);b(17)=d(17)^(1/17);
beta_min15 = max(b(16:17));
s15=max(0,ceil(log2(beta_min15/Theta(6))/2));
if s12<=s15
    d(16)=min(d(16),norm1pp(pB{3},5,pB{1}));b(16)=d(16)^(1/16);
    s15aux=max(0,ceil(log2(b(16)/Theta(6))/2)); %Temptative scaling for m=15
    if s15aux<s12    
        d(17)=min(d(17),d(16)*b(1));
        b(17)=d(17)^(1/17);
        beta_min15=min(beta_min15,max(b(16:17)));
        s15=max(0,ceil(log2(beta_min15/Theta(6))/2));
        if s15<s12
            m=15; % q=3
            s=s15; 
            return
        end  %If m=15 has a lower scaling s15 and the same or less cost than m=12, m=15 is preferred
        d(17)=norm1pp(pB{3},5,pB{2});
        b(17)=d(17)^(1/17);
        beta_min15=min(beta_min15,max(b(16:17)));
        s15=max(0,ceil(log2(beta_min15/Theta(6))/2));
        if s15<s12
            m=15; % q=3
            s=s15; 
            return
        else
            m=12; % q=3
            s=s12; 
            return
        end %If m=15 has a lower scaling s15 and the same or less cost than m=12, m=15 is preferred
    else
        m=12; % q=3
        s=s12; 
        return
    end %If m=12 has less cost than m=15, m=12 is preferred
else
    m=15; % q=3
    s=s15; 
    return
end
error('selecciona_ms_cos_taylor_sastre_conEstNorma:NoParam','Cannot find valid parameters, check matrix')
end