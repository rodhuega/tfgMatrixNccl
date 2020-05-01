function [coef,potA] = coefs_cosh_bernoulli_sym(m,formulacion)
% [coef,potA] = coefs_cosh_bernoulli_sym(m,formulacion)
% Coeficientes y potencias de la aproximaci�n de Bernoulli en el c�lculo de 
% cosh(A) calculados de forma simb�lica.
%
% Datos de entrada:
% - m:           Orden de la aproximaci�n, de modo que el grado del 
%                polinomio a emplear ser� 2*m.
% - formulaci�n: Formulaci�n te�rica para trabajar s�lo con los t�rminos de 
%                posiciones pares ('pares') o con todos los t�rminos
%                ('pares_impares').
%
% Datos de salida:
% - coef:        Coeficientes de la forma simb�lica del polinomio de grado 
%                m.
% - potA:        Potencias de A del polinomio de grado m.

syms A;
% Obtenemos el polinomio de forma simb�lica
E=cosh_bernoulli(A,m,formulacion);
% Separamos los coeficientes y las potencias
[coef,potA]=coeffs(collect(E,A),A);
coef=coef(end:-1:1);
potA=potA(end:-1:1);
