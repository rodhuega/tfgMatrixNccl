function [fA,m,s,np] = cosmber_3_4(A)
% [fA,m,s,np] = cosmber_3_4(A)
%
% C�lculo del coseno matricial por Bernoulli mediante:
%   - 'terminos_pares_polinomio_solo_pares'.
%   - 'conEstNormaConPotenciasNuevo'.
%
% Datos de entrada:
% - A:     Matriz de la cual calculamos f(A).
%
% Datos de salida:
% - fA:    Valor de la funci�n f sobre la matriz A.
% - m:     Grado del polinomio empleado en la aproximaci�n.
% - s:     Valor del escalado de la matriz.
% - nProd: N�mero de productos matriciales involucrados para calcular f(A).

formulacion_seleccion_ms(1)=3;
formulacion_seleccion_ms(2)=5;
[fA,m,s,np]=cosmber(A,formulacion_seleccion_ms);
end