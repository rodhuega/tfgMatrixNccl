function g=series_absolute_back_taylor(f,m,n)
%f=coef_back(f,m,itermax)
%This function computes  the first  aproximately n terms of the
%corresponding series of the absolute backward error of function f(x).
%Input Data
%  f: Analytic function.
%  m: Positive integer equal to the order of the Taylor approximation.
%  n: Positive integer. The number of the computed terms of backward error 
%     is equal to the first multiple of m+1 greater or equal to n.
%Output Data
%  e_backward: vector with aproximately the first n coefficients of 
%     backward error.
%Example 1: MATLAB Built-in functions
%e_backward=coef_back(@exp,4,100)
%
%Example 2: General MATLAB  functions
%If we define the following functions as a MATLAB function
% function f= ej(x)
% f=sin(x)*exp(x);
% end
%Then
%e_backward=coef_back(@ej,4,100)
syms x;
g=series_back_taylor(f,m,n);
[C,X]=coeffs(subs(g,x));
g=abs(C)*X.';
