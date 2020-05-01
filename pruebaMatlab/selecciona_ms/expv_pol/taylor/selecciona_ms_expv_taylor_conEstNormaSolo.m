function [m,s,pA,nProd]=selecciona_ms_expv_taylor_conEstNormaSolo(plataforma,A,mmax)
% [m,s,pA,nProd]=selecciona_ms_expv_taylor_conEstNormaSolo(plataforma,A,mmax)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz, con estimaciones de la norma de las potencias 
% matriciales, para calcular la acci�n de la funci�n exponencial mediante
% Taylor.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la funci�n matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
% - mmax:       Valor m�ximo del grado del polinomio de aproximaci�n.
%               Valores posibles son 2, 4, 6, 9, 12, 16, 20, 25, 30.
%
% Datos de salida:
% - m:          Orden de la aproximaci�n polin�mica a f(A). Coincide con el 
%               grado del polinomio de aproximaci�n a f(A).
% - s:          Valor del escalado de la matriz.
% - pA:         Array de celdas de una �nica componente con la matriz A.
% - nProd:      N�mero de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

% PENDIENTE DE MODIFICAR
[m,s,P,nProd]=selecciona_ms_exp_taylor_conEstNormaSolo(plataforma,A,mmax);
pA{1}=A;
end

