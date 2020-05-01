function [coef,potA] = coefs_exp_bernoulli_sym(m,L)
% [coef,potA] = coefs_exp_bernoulli_sym(m,L)
% Coeficientes y potencias de la aproximaci�n de Bernoulli en el c�lculo de 
% e^A calculados de forma simb�lica.
%
% Datos de entrada:
% - m: Orden de la aproximaci�n. Coincide con el grado del polinomio a 
%      emplear (2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 49, 56, 64, ...).
% - L: Lambda o escalado de la variable t. Puede venir dado de forma 
%      num�rica o simb�lica.
%
% Datos de salida:
% - coef: Coeficientes de la forma simb�lica del polinomio de grado m.
% - potA: Potencias de A del polinomio de grado m.

syms A;
% Obtenemos el polinomio de forma simb�lica
E=exp_bernoulli(A,m,L);
% Separamos los coeficientes y las potencias
[coef,potA]=coeffs(collect(E,A),A);
coef=coef(end:-1:1);
potA=potA(end:-1:1);

