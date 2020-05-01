function [coef,potA] = coefs_cosh_hermite_sym(m)
% [coef,potA] = coefs_cosh_hermite_sym(m)
% Coeficientes y potencias de la aproximación de Hermite en el cálculo de 
% cosh(A) calculados de forma simbólica.
%
% Datos de entrada:
% - m: Orden de la aproximación.
%
% Datos de salida:
% - coef: Coeficientes de la forma simbólica del polinomio de grado m.
% - potA: Potencias de A del polinomio de grado m.

syms A;
% Obtenemos el polinomio de forma simbólica
E=cosh_hermite(A,m);
% Separamos los coeficientes y las potencias
[coef,potA]=coeffs(collect(E,A),A);
coef=coef(end:-1:1);
potA=potA(end:-1:1);
