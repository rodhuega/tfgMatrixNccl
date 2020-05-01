function [g] = series_absolute_forward_error(f,m,nterm)
% Serie de Taylor del error relativo forward de la funci�n f
%m: orden de la aproximaci�n de Taylor
%n: n�mero t�rminos de la serie del error absoluto 
syms x
g = taylor(f(x)-taylor(f(x),'order',m+1),'order',m + nterm+1);
[C,X]=coeffs(g);
g=abs(C)*X.';
