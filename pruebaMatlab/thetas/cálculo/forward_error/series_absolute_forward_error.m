function [g] = series_absolute_forward_error(f,m,nterm)
% Serie de Taylor del error relativo forward de la función f
%m: orden de la aproximación de Taylor
%n: número términos de la serie del error absoluto 
syms x
g = taylor(f(x)-taylor(f(x),'order',m+1),'order',m + nterm+1);
[C,X]=coeffs(g);
g=abs(C)*X.';
