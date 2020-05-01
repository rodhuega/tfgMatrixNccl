function [coef,potA] = coefs_exp_taylor_sym(m)
% [coef,potA] = coefs_exp_taylor_sym(m)
% Coeficientes y potencias de la aproximaci�n de Taylor en el c�lculo de 
% e^A calculados de forma simb�lica.
%
% Datos de entrada:
% - m: Orden de la aproximaci�n. Coincide con el grado del polinomio a 
%      emplear (2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 49, 56, 64, ...).
%
% Datos de salida:
% - coef: Coeficientes de la forma simb�lica del polinomio de grado m.
% - potA: Potencias de A del polinomio de grado m.

syms A;
% Obtenemos el polinomio de forma simb�lica
E=exp_taylor(A,m);
% Separamos los coeficientes y las potencias
[coef,potA]=coeffs(collect(E,A),A);
coef=coef(end:-1:1);
potA=potA(end:-1:1);
