function [m,s,pA,nProd]=selecciona_ms_expv_taylor_conEstNormaSolo(plataforma,A,mmax)
% [m,s,pA,nProd]=selecciona_ms_expv_taylor_conEstNormaSolo(plataforma,A,mmax)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz, con estimaciones de la norma de las potencias 
% matriciales, para calcular la acción de la función exponencial mediante
% Taylor.
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
% - pA:         Array de celdas de una única componente con la matriz A.
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

% PENDIENTE DE MODIFICAR
[m,s,P,nProd]=selecciona_ms_exp_taylor_conEstNormaSolo(plataforma,A,mmax);
pA{1}=A;
end

