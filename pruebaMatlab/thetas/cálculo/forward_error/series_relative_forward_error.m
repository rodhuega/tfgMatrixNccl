function [g] = series_relative_forward_error(f,m,nterm)
% Serie de Taylor del error relativo forward de la función f
%m: orden de la aproximación de Taylor
%n: número términos de la serie del error absoluto 
syms x
g = taylor(1-taylor(f(x),'order',m+1)/f(x),'order',m + nterm+1);
[C,X]=coeffs(subs(g,x));
g=abs(C)*X.';
