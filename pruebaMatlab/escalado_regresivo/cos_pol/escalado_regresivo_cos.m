function [A,nProd] = escalado_regresivo_cos(plataforma,A,s)
% [A,nProd] = escalado_regresivo_cos(plataforma,A,s)
%
% T�cnica de squaring de la matriz A tras haber aplicado la funci�n
% coseno o coseno hiperb�lico.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la funci�n matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz resultado de la evaluaci�n del polinomio de 
%               aproximaci�n.
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - A:          Resultado tras la t�cnica de squaring.
% - nProd:      N�mero de productos matriciales que se han llevado a cabo
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
% T�cnica de squaring de la matriz A, tras haber aplicado la funci�n
% coseno o coseno hiperb�lico, sin emplear GPUs.
%
% Datos de entrada:
% - A:     Matriz resultado de la evaluaci�n del polinomio de aproximaci�n.
% - s:     Valor del escalado de la matriz.

%
% Datos de salida:
% - A:     Resultado tras la t�cnica de squaring.
% - nProd: N�mero de productos matriciales que se han llevado a cabo
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
% T�cnica de squaring de la matriz A, tras haber aplicado la funci�n
% coseno o coseno hiperb�lico, mediante GPUs.
%
% Datos de entrada:
% - s:     Valor del escalado de la matriz.
%
% Datos de salida:
% - nProd: N�mero de productos matriciales que se han llevado a cabo
%          con el reescalado.

call_gpu('unscale',s);

nProd=s;
end
