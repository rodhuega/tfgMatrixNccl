function [coef,potA] = coefs_cosh_bernoulli_sym(m,formulacion)
% [coef,potA] = coefs_cosh_bernoulli_sym(m,formulacion)
% Coeficientes y potencias de la aproximación de Bernoulli en el cálculo de 
% cosh(A) calculados de forma simbólica.
%
% Datos de entrada:
% - m:           Orden de la aproximación, de modo que el grado del 
%                polinomio a emplear será 2*m.
% - formulación: Formulación teórica para trabajar sólo con los términos de 
%                posiciones pares ('pares') o con todos los términos
%                ('pares_impares').
%
% Datos de salida:
% - coef:        Coeficientes de la forma simbólica del polinomio de grado 
%                m.
% - potA:        Potencias de A del polinomio de grado m.

syms A;
% Obtenemos el polinomio de forma simbólica
E=cosh_bernoulli(A,m,formulacion);
% Separamos los coeficientes y las potencias
[coef,potA]=coeffs(collect(E,A),A);
coef=coef(end:-1:1);
potA=potA(end:-1:1);
