function [fAv,m,s,np] = expmvtay3(A,v)
% [fA,m,s,np] = expmvtay3(A,v)
%
% Cálculo de la accion exponencial matricial por Taylor.
%
% Datos de entrada:
% - A:            Matriz de la cual calculamos f(A).
% - v:            Vector para calcular f(A)*v.
%
% Datos de salida:
% - fAv:          Valor de la acción f(A)*v.
% - m:            Grado del polinomio empleado en la aproximación.
% - s:            Valor del escalado de la matriz.
% - nProd:        Número de productos matriciales involucrados para calcular
%                 f(A).
seleccion_ms=3;
[fAv,m,s,np] = expmvtay(A,v,seleccion_ms);
end