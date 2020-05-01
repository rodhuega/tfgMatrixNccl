function [fA,m,s,np] = cosmber_2_5(A)
% [fA,m,s,np] = cosmber_2_5(A)
%
% Cálculo del coseno matricial por Bernoulli mediante:
%   - 'terminos_pares_polinomio_completo'.
%   - 'conEstNormaSinPotencias'.
%
% Datos de entrada:
% - A:     Matriz de la cual calculamos f(A).
%
% Datos de salida:
% - fA:    Valor de la función f sobre la matriz A.
% - m:     Grado del polinomio empleado en la aproximación.
% - s:     Valor del escalado de la matriz.
% - nProd: Número de productos matriciales involucrados para calcular f(A).

formulacion_seleccion_ms(1)=1;
formulacion_seleccion_ms(2)=3;
[fA,m,s,np]=cosmber(A,formulacion_seleccion_ms);
end