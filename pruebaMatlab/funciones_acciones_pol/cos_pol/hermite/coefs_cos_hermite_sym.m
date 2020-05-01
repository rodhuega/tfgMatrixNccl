function [coef,potA] = coefs_cos_hermite_sym(m)
% [coef,potA] = coefs_cos_hermite_sym(m)
% Coeficientes y potencias de la aproximaci�n de Hermite en el c�lculo de 
% cos(A) calculados de forma simb�lica.
%
% Datos de entrada:
% - m: Orden de la aproximaci�n.
%
% Datos de salida:
% - coef: Coeficientes de la forma simb�lica del polinomio de grado m.
% - potA: Potencias de A del polinomio de grado m.

syms A;
% Obtenemos el polinomio de forma simb�lica
E=cos_hermite(A,m);
% Separamos los coeficientes y las potencias
[coef,potA]=coeffs(collect(E,A),A);
coef=coef(end:-1:1);
potA=potA(end:-1:1);
