function [m,sm,pA,nProd]=selecciona_ms_cosh_bernoulli_conEstNormaSolo(plataforma,A,mmax)
% [m,sm,pA,nProd]=selecciona_ms_cosh_bernoulli_conEstNormaSolo(plataforma,A,mmax)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz, de forma novedosa y exclusivamente con estimaciones de 
% la norma de las potencias matriciales, para aplicar la funci�n coseno
% hiperb�lico sobre A mediante Bernoulli.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la funci�n matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
% - mmax:       Valor m�ximo del grado del polinomio de aproximaci�n.
%               Valores posibles son 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42
%               49, 56, 64.
%
% Datos de salida:
% - m:          Orden de la aproximaci�n polin�mica a f(A). Coincide con el 
%               grado del polinomio de aproximaci�n a f(A).
% - sm:         Valor del escalado de la matriz.
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i}
%               contiene A^i, para i=1,2,3,...,q, siendo q=ceil(sqrt(m)).
% - nProd:      N�mero de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

%zeta=[
%      0.0065633223103254337  % m=2  forward bound
%      0.11495105955344324    % m=4  forward bound
%      0.43834831618193604    % m=6  forward bound
%      1.3228006323567987     % m=9  forward bound
%      2.5674905431377995     % m=12 forward bound
%      4.5920603879163053     % m=16 forward bound
%      6.8812790938356159     % m=20 forward bound
%      9.9720273582143477     % m=25 forward bound
%      13.223807251049353     % m=30 forward bound
%      17.260373202087322     % m=36 forward bound
%      21.392801746036426     % m=42 forward bound
%      26.293796454194702     % m=49 forward bound
%      31.253166710014156     % m=56 forward bound
%      36.970006057142768];   % m=64 forward bound
 
% VALORES DE ZETA PENDIENTES DE CALCULAR POR JAVIER

 zeta=[0.00053172328233359433 % m=2  backward bound
       0.066928018242067802   % m=4  backward bound
       0.38132221513672399    % m=6  backward bound
       1.3410838441039987     % m=9  backward bound
       2.5985282387162836     % m=12 backward bound
       3.4269238450658972     % m=16 backward bound
       3.9534599106168673     % m=20 backward bound
       4.7751891209499346     % m=25 backward bound
       5.8357027885345474     % m=30 backward bound
       7.5250770772072215     % m=36 backward bound
       9.8289020699016394     % m=42 backward bound
       13.612347620781113     % m=49 backward bound
       19.103151143908487     % m=56 backward bound
       28.541101502535394];   % m=64 backward bound 

zeta=sqrt(zeta);
   
M = [2 4 6 9 12 16 20 25 30 36 42 49 56 64];
pot=[2 2 3 3  4  4  5  5  6  6  7  7  8  8];

% Buscamos la posici�n (imax) de mmax en el vector M
i=1;
encontrado=0;
while i<=length(M) && encontrado==0
    if M(i)==mmax
        imax=i;
        encontrado=1;
    else
        i=i+1;
    end
end
if (encontrado==0)
    error('Valor mmax no permitido (emplear 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 49, 56 o 64)');
end

tol=1e-2;
alpham=compute_alpha(A,tol,imax);
nofin=1;
im=0;
while nofin && im<imax
    im=im+1;
    if alpham<zeta(im)
        nofin=0;     
    end
end
if nofin==0
    sm=0;
else
    sm=ceil(max(0,log2(alpham/zeta(im))));
    j=im;
    fin=0;
    while fin==0
        j=j-1;
        s=ceil(max(0,log2(alpham/zeta(j))));
        if sm>=s
            sm=s;
            im=j;
        else
            fin=1;
        end
    end
end 

m=M(im);

% C�lculo de las potencias de A
q=pot(im);
nProd=0;
switch plataforma
    case 'sinGPUs'
        pA = cell(q,1);
        pA{1}=A;
        for i=2:q
            pA{i}=pA{i-1}*A;
            nProd=nProd+1;
        end
    case 'conGPUs'
        pA=[];
        for i=2:q
            call_gpu('power');
            nProd=nProd+1;
        end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [alpha,i]=compute_alpha(A,tol,imax)
M= [2 4 6 9 12 16 20 25 30 36 42 49 56 64];
nofin=1;
i=1;
p=M(i)+1;
alpha0=norm1p(A,p);
while nofin && i<imax
    i=i+1;p=M(i)+1;
    alpha=norm1p(A,p)^(1/p);
    if (alpha-alpha0)/alpha>tol
        alpha=alpha0;
    else
        nofin=0;
    end
end
end




