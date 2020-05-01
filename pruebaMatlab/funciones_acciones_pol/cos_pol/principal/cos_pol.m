function [fA,nProd]=cos_pol(metodo_f,plataforma,pB,m,s)
% [fA,nProd]=cos_pol(metodo_f,plataformapB,m,s)
% Funcion principal para el cálculo del coseno de una matriz.
%
% Datos de entrada:
% - metodo_f:   Método a emplear para calcular el coseno (taylor, hermite,
%               bernoulli, etc.).
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - pB:         Vector de cells arrays con las potencias de B, de modo que
%               pB{i} contiene B^i, para i=1,2,3,...,q, siendo q<=4, B=A*A.
%               Como excepción, con el método de Bernoulli será un vector
%               de celdas con las potencias de A, de modo que pB{i}
%               contendrá A^i, para i=1,2,3,...,q, siendo q=ceil(sqrt(m)).
% - m:          Orden de la aproximación polinómica a f(A). El grado del 
%               polinomio de aproximación a f(A) será 2*m.
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - fA:         Valor de la función coseno sobre la matriz A.
% - nProd.      Número de productos matriciales llevados a cabo para 
%               calcular el coseno de A.

switch metodo_f
    case 'taylor'
        p=coefs_cos_taylor(m);
    case 'bernoulli'       
        p=coefs_cos_bernoulli(m);
    case 'taylor_bernoulli'
        if m<30
            p=coefs_cos_taylor(m);
        else
            p=coefs_cos_bernoulli(m);
        end
    case 'taylor_sastre'
        p=coefs_cos_taylor_sastre(m);        
    case 'hermite'
        p=coefs_cos_hermite(m);
    otherwise
        error('Método no válido para calcular el coseno');
end

% Escalado
pB=escalado('cos',metodo_f,plataforma,pB,s);

% Evaluamos la expresión polinómica de forma eficiente
switch metodo_f
    case 'taylor_sastre'
        % Por Sastre
        [fA,nProd_eval] = polyvalm_cos_taylor_sastre(plataforma,p,pB);
    otherwise
        % Por Paterson-Stockmeyer
        [fA,nProd_eval] = polyvalm_paterson_stockmeyer(plataforma,p,pB);
end

% Reescalado
[fA,nProd_er]=escalado_regresivo('cos',plataforma,fA,s);

% Número final de productos
nProd=nProd_eval+nProd_er;

end
