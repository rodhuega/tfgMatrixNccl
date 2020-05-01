function [A,nProd] = escalado_regresivo_cos(plataforma,A,s)
% [A,nProd] = escalado_regresivo_cos(plataforma,A,s)
%
% Técnica de squaring de la matriz A tras haber aplicado la función
% coseno o coseno hiperbólico.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz resultado de la evaluación del polinomio de 
%               aproximación.
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - A:          Resultado tras la técnica de squaring.
% - nProd:      Número de productos matriciales que se han llevado a cabo
%               con el reescalado.

switch plataforma
    case 'sinGPUs'
        [A,nProd]=escalado_regresivo_cos_sinGPUs(A,s);
    case 'conGPUs'
        nProd=escalado_regresivo_cos_conGPUs(s);
    otherwise
        error('Plataforma destino incorrecta');
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A,nProd] = escalado_regresivo_cos_sinGPUs(A,s)
% [A,nProd] = escalado_regresivo_cos_sinGPUs(A,s)
%
% Técnica de squaring de la matriz A, tras haber aplicado la función
% coseno o coseno hiperbólico, sin emplear GPUs.
%
% Datos de entrada:
% - A:     Matriz resultado de la evaluación del polinomio de aproximación.
% - s:     Valor del escalado de la matriz.

%
% Datos de salida:
% - A:     Resultado tras la técnica de squaring.
% - nProd: Número de productos matriciales que se han llevado a cabo
%          con el reescalado.

n=size(A,1);
I=eye(n);
% Recovering cos(A) o cosh(A) from the scaled aproximation cos(2^(-s)*A)
for i = 1:s % double angle phase
    A = 2*A*A-I;  
end

nProd=s;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nProd = escalado_regresivo_cos_conGPUs(s)
% nProd = escalado_regresivo_cos_conGPUs(s)
%
% Técnica de squaring de la matriz A, tras haber aplicado la función
% coseno o coseno hiperbólico, mediante GPUs.
%
% Datos de entrada:
% - s:     Valor del escalado de la matriz.
%
% Datos de salida:
% - nProd: Número de productos matriciales que se han llevado a cabo
%          con el reescalado.

call_gpu('unscale',s);

nProd=s;
end
