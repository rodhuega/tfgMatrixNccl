function [m,s,pA,nProd]=selecciona_ms_exp_taylor_conEstNorma(plataforma,A,mmax)
% [m,s,pA,nProd]=selecciona_ms_exp_taylor_conEstNorma(plataforma,A,mmax)
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
%               Valores posibles son 2, 4, 6, 9, 12, 16, 20, 25, 30.
%
% Datos de salida:
% - m:          Orden de la aproximación polinómica a f(A). Coincide con el 
%               grado del polinomio de aproximación a f(A).
% - s:          Valor del escalado de la matriz.
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i}
%               contiene A^i, para i=1,2,3,...,q, siendo q=ceil(sqrt(m)) o
%               q=floor(sqrt(m)).
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

Thetam1=1.490116111983279e-8; %m=1
theta=[8.733457513635361e-006 %m=2
       1.678018844321752e-003 %m=4
       1.773082199654024e-002 %m=6
       1.137689245787824e-001 %m=9
       3.280542018037257e-001 %m=12
       7.912740176600240e-001 %m=16
       1.438252596804337      %m=20
       2.428582524442827      %m=25
       3.539666348743690];    %m=30

c=[   4/3   %m=2   %c=abs(c_m(m+1)/c_m(m+2))
      6/5   %m=4   
      8/7   %m=6
    11/10   %m=9
    14/13   %m=12
    18/17   %m=16
    22/21   %m=20
    27/26   %m=25
    32/31]; %m=30

ucm2= [8.9e-016                %m=2 
       1.598721155460225e-014  %m=4 
       6.394884621840902e-013  %m=6
       4.431655042935745e-010  %m=9
       7.445180472132051e-007  %m=12
       4.181213353149360e-002  %m=16
       5.942340417495871e+003  %m=20
       4.649643683073819e+010  %m=25
       9.423674633957229e+017];%m=30

mv = [2 4 6 9 12 16 20 25 30];  %Vector of optimal orders m.

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
if (encontrado==0)
    error('Valor mmax no permitido (emplear 2, 4, 6, 9, 12, 16, 20, 25, 30)');
end

if mmax>20 %m_M = 25 or 30
    qv = [2  2  3  3  4  4  5  5  5];
else           %m_M = 16 or 20
    qv = [2  2  3  3  4  4  4];
end

% Una alternativa al vector anterior es la siguiente:
% qv = [2  2  3  3  4  4  5  5  6];
% Si se opta por esta alternativa, hay que descomentar las instrucciones de
% la 212 a la 218 del case 30 y, tal vez, cambiar las instrucciones 220 y 
% 223.
%for i=1:length(mv)
%    qv(i)=ceil(sqrt(mv(i)));
%end

s = 0;
nProd = 0;
qmax = qv(kmax);

a=-ones(1,mmax+2);
%pA = cell(qmax,1);
pA{1}=A;
switch plataforma
    case 'sinGPUs'
        a(1)=norm(pA{1},1);
    case 'conGPUs'
        a(1)=call_gpu('norm1',1);
end
if a(1)<=Thetam1            % m=1 not very likely to be used
    m = 1;                            
    return
end

for inm=1:kmax
    m = mv(inm);
    switch m
        case 2        
            switch plataforma
                case 'sinGPUs'
                    pA{2} = pA{1}*A;
                case 'conGPUs'
                    pA{2}=call_gpu('power');
            end
            nProd = nProd+1;
            a(3) = norm1pp(pA{2},1,A);
            b = max(1,a(1))*ucm2(1);
            if c(1)*a(3)<=b
                a(4) = norm1p(pA{2},2);
                if c(1)*a(3)+a(4)<=b 
                    return
                end
            end

        case 4
            a(5) = norm1pp(pA{2},2,A);
            b = max(1,a(1))*ucm2(2);
            if c(2)*a(5)<=b
                a(6) = norm1p(pA{2},3);
                if c(2)*a(5)+a(6)<=b 
                    return
                end
            end
            
        case 6
            switch plataforma
                case 'sinGPUs'
                    pA{3} = pA{2}*A;
                case 'conGPUs'
                    pA{3}=call_gpu('power');
            end            
            nProd = nProd+1;
            a(7) = norm1pp(pA{3},2,A);  
            b = max(1,a(1))*ucm2(3);
            if c(3)*a(7)<=b
                a(8) = norm1pp(pA{3},2,pA{2});
                if c(3)*a(7)+a(8)<=b 
                    return
                end
            end
            
        case 9
            a(10) = norm1pp(pA{3},3,A);
            b = max(1,a(1))*ucm2(4);
            if c(4)*a(10)<=b
                a(11) = norm1pp(pA{3},3,pA{2});
                if c(4)*a(10)+a(11)<=b 
                    return
                end
            end
            
        case 12
            switch plataforma
                case 'sinGPUs'
                    pA{4} = pA{3}*A;
                case 'conGPUs'
                    pA{4}=call_gpu('power');
            end            
            nProd = nProd+1;
            a(13) = norm1pp(pA{4},3,A);  
            b = max(1,a(1))*ucm2(5);
            if c(5)*a(13)<=b
                a(14) = norm1pp(pA{4},3,pA{2});
                if c(5)*a(13)+a(14)<=b 
                    return
                end
            end
            
        case 16
            a(17) = norm1pp(pA{4},4,A);
            b = max(1,a(1))*ucm2(6);
            if c(6)*a(17)<=b
                a(18) = norm1pp(pA{4},4,pA{2});
                if c(6)*a(17)+a(18)<=b 
                    return
                end
            end
            
        case 20
            switch plataforma
                case 'sinGPUs'
                    pA{5} = pA{4}*A;
                case 'conGPUs'
                    pA{5}=call_gpu('power');
            end 
            nProd = nProd+1;
            a(21) = norm1pp(pA{5},4,A);  
            b = max(1,a(1))*ucm2(7);
            if c(7)*a(21)<=b
                a(22) = norm1pp(pA{5},4,pA{2});
                if c(7)*a(21)+a(22)<=b 
                    return
                end
            end

        case 25
            a(26) = norm1pp(pA{5},5,A);  
            b = max(1,a(1))*ucm2(8);
            if c(8)*a(26)<=b
                a(27) = norm1pp(pA{5},5,pA{2});
                if c(8)*a(26)+a(27)<=b 
                    return
                end
            end
            
        case 30
            %switch plataforma
            %    case 'sinGPUs'
            %        pA{6} = pA{5}*A;
            %    case 'conGPUs'
            %        pA{6}=call_gpu('power');
            %end
            %nProd = nProd+1;
            a(31) = norm1pp(pA{5},6,A);  
            b = max(1,a(1))*ucm2(9);
            if c(9)*a(31)<=b
                a(32) = norm1pp(pA{5},6,pA{2});
                if c(9)*a(31)+a(32)<=b 
                    return
                end
            end         
    end
end

% Estimate 1-norm of A^(mmax+2) if it is not already estimated
if a(mmax+2)<0
    a(mmax+2) = norm1pp(pA{qmax},floor((mmax+2)/qmax),pA{mod((mmax+2),qmax)});
end

% Compute alpha_min
normA = max(a(mmax+1)^(1/(mmax+1)), a(mmax+2)^(1/(mmax+2)));
[t, s] = log2(normA/theta(kmax));
s = s - (t == 0.5); % adjust s if normA/theta(end) is a power of 2.

% Test if s can be reduced, checking if (5) of [2] holds with s=s-1
if s>0 
    s = s-1;
    b = max(1,a(1)/2^s)*ucm2(kmax);
    if c(kmax)*a(mmax+1)/2^((mmax+1)*s)+a(mmax+2)/2^((mmax+2)*s)>b
        % Cannot reduce s
        s = s+1;
    end
end

% Test if scaled matrix allows using mv(kmax-1)
mmax = mv(kmax-1);
qmax = qv(kmax-1);

% Estimate 1-norm of A^(mmax+2) if it is not already estimated
if a(mmax+2)<0
    a(mmax+2) = norm1pp(pA{qmax},floor((mmax+2)/qmax),pA{mod((mmax+2),qmax)});
end

b = max(1,a(1)/2^s)*ucm2(kmax-1);
if c(kmax-1)*a(mmax+1)/2^((mmax+1)*s)+a(mmax+2)/2^((mmax+2)*s)<=b
    m = 25;  % Scaled matrix allows using mv(kmax-1)
end

end

