function [fAv,m,s,np] = expmvtay_inef2(A,v)
% [fA,m,s,np] = expmvtay_inef2(A,v)
%
% Cálculo de la accion exponencial matricial por Taylor.
%
% Datos de entrada:
% - A:            Matriz de la cual calculamos f(A).
% - v:            Vector para calcular f(A)*v..
%
% Datos de salida:
% - fAv:          Valor de la acción f(A)*v.
% - m:            Grado del polinomio empleado en la aproximación.
% - s:            Valor del escalado de la matriz.
% - nProd:        Número de productos matriciales involucrados para calcular
%                 f(A).

seleccion_ms=2;
[fAv,m,s,np] = expmvtay_inef(A,v,seleccion_ms);
end