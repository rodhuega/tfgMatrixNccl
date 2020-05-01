function [g] = series_relative_forward_error(f,m,nterm)
% Serie de Taylor del error relativo forward de la funci�n f
%m: orden de la aproximaci�n de Taylor
%n: n�mero t�rminos de la serie del error absoluto 
syms x
g = taylor(1-taylor(f(x),'order',m+1)/f(x),'order',m + nterm+1);
[C,X]=coeffs(subs(g,x));
g=abs(C)*X.';
