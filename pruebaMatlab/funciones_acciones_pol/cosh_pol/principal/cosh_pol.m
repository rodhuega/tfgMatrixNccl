function [fA,nProd]=cosh_pol(metodo_f,plataforma,pB,m,s)
% [fA,nProd]=cosh_pol(metodo_f,plataforma,pB,m,s)
% Funcion principal para el cálculo del coseno hiperbólico de una matriz.
%
% Datos de entrada:
% - metodo_f:   Método a emplear para calcular el coseno hiperbólico 
%               (taylor, hermite, bernoulli, etc.).
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - pB:         Vector de cells arrays con las potencias de B, de modo que
%               pB{i} contiene B^i, para i=1,2,3,...,q, siendo q<=4, B=A*A.
% - m:          Orden de la aproximación polinómica a f(A). El grado del 
%               polinomio de aproximación a f(A) será 2*m.
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - fA:         Valor de la función coseno hiperbólico sobre la matriz A.
% - nProd.      Número de productos matriciales llevados a cabo para 
%               calcular el coseno hiperbólico de A.

switch metodo_f
    case 'taylor'
        %p=coefs_cosh_taylor(m);
    case 'bernoulli'
        formulacion='pares';
        %formulacion='pares_impares';
        p=coefs_cosh_bernoulli(m,formulacion);        
    case 'hermite'
        p=coefs_cosh_hermite(m);
    otherwise
        error('Método no válido para calcular el coseno hiperbólico');
end

% Escalado
pB=escalado('cosh',metodo_f,plataforma,pB,s);

% Evaluamos la expresión polinómica de forma eficiente
switch metodo_f
    case {'taylor','bernoulli','hermite'}
        % Por Paterson-Stockmeyer
        [fA,nProd_eval] = polyvalm_paterson_stockmeyer(plataforma,p,pB);
end

% Reescalado
[fA,nProd_er]=escalado_regresivo('cosh',plataforma,fA,s);

% Número final de productos
nProd=nProd_eval+nProd_er;

end
