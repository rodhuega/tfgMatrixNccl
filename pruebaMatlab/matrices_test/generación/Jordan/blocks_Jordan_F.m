function [A,FA]=blocks_Jordan_F(F,v,n,nd)
% [A,FA]=blocks_Jordan_F(F,v,n,nd)
%
% Datos de entrada:
% - F:  Función a aplicar sobre la matriz (@exp, @cos, @cosh, ...).
% - v:  Vector de valores propios.
% - n:  Vector de multiplicidades.
% - nd: Número de dígitos vpa (32, 64, 128, 256).
%
% Datos de salida:
% - A:  Matriz generada.
% - FA: Resultado de la función matricial aplicada sobre la matriz A.

digits(nd);
m=length(v);
i=1;
for j=1:m
    [J,eJ]=block_jordan_F(F,v(j),n(j),nd);
    A(i:i+n(j)-1,i:i+n(j)-1)=J;
    FA(i:i+n(j)-1,i:i+n(j)-1)=eJ;
    i=i+n(j);
end