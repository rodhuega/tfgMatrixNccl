function [fA,nProd]=exp_pol(metodo_f,plataforma,pA,m,s)
% [fA,nProd]=exp_pol(metodo_f,plataforma,pA,m,s)
%
% Funcion principal para el c�lculo de la exponencial de una matriz.
%
% Datos de entrada:
% - metodo_f:   M�todo a emplear para calcular la exponencial (taylor, 
%               bernoulli, hermite, ...).
% - plataforma: Decide si calculamos la funci�n matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - pA:         Vector de cells arrays con las potencias de A, de modo que
%               pA{i} contiene A^i, para i=1,2,3,...,q, siendo 
%               q=ceil(sqrt(m))o q=floor(sqrt(m)).
% - m:          Orden de la aproximaci�n polin�mica a f(A). 
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - fA:         Valor de la funci�n exponencial sobre la matriz A.
% - nProd.      N�mero de productos matriciales llevados a cabo para 
%               calcular la exponencial de A.

switch metodo_f
    case {'taylor','splines'}
        p=coefs_exp_taylor(m);
    case 'bernoulli'
        L=1;
        p=coefs_exp_bernoulli(m,L);
    case 'taylor_bernoulli'
        switch m
            case {25,30}
                L=1;
                p = coefs_exp_bernoulli(m,L);
            otherwise
                p = coefs_exp_taylor(m);
        end
    case 'taylor_sastre'
        p=coefs_exp_taylor_sastre(m);
    otherwise
        error('M�todo no v�lido para calcular la exponencial');
end

% Escalado
pA=escalado('exp',metodo_f,plataforma,pA,s);

% Evaluamos la expresi�n polin�mica de forma eficiente
switch metodo_f
    case 'taylor_sastre'
        % Por Sastre
        [fA,nProd_eval] = polyvalm_exp_taylor_sastre(plataforma,p,pA);
    case 'splines'
        % Por splines y por Paterson-Stockmeyer
        [fA,nProd_eval] = polyvalm_exp_splines(plataforma,p,pA);
    otherwise
        % Por Paterson-Stockmeyer
        [fA,nProd_eval] = polyvalm_paterson_stockmeyer(plataforma,p,pA);
end

% Reescalado
[fA,nProd_er]=escalado_regresivo('exp',plataforma,fA,s);

% N�mero final de productos
nProd=nProd_eval+nProd_er;
end
