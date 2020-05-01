function [coef,potA] = coefs_cos_bernoulli_sym(m)
% [coef,potA] = coefs_cos_bernoulli_sym(m)
% Coeficientes y potencias de la aproximación de Bernoulli en el cálculo de 
% cos(A) calculados de forma simbólica.
%
% Datos de entrada:
% - m: Orden de la aproximación, de modo que el grado del polinomio a 
%      emplear será 2*m.
%
% Datos de salida:
% - coef:        Coeficientes de la forma simbólica del polinomio de grado 
%                m.
% - potA:        Potencias de A del polinomio de grado m.

syms A;
% Obtenemos el polinomio de forma simbólica
E=cos_bernoulli(A,m);
% Separamos los coeficientes y las potencias
[coef,potA]=coeffs(collect(E,A),A);
coef=coef(end:-1:1);
potA=potA(end:-1:1);
