function fAv=expv_pol(metodo_f,plataforma,A,v,m,s)
% fAv=expv_pol(metodo_f,plataforma,A,v,m,s)
%
% Funcion principal para el cálculo de la acción de la exponencial de una 
% matriz.
%
% Datos de entrada:
% - metodo_f:   Método a emplear para calcular la exponencial (taylor, 
%               bernoulli, hermite, ...).
% - plataforma: Decide si calculamos la acción matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz a calcular su exponencial expresada en forma de un
%               array de celdas de una única componente.
% - v:          Vector columna para calcular el valor de la acción de la 
%               función matricial, es decir, f(A)*v sin obtener f(A) de 
%               forma explícita. 
% - m:          Orden de la aproximación polinómica a f(A). 
% - s:          Valor del escalado de la matriz A.
%
% Datos de salida:
% - fA:         Valor de la acción de la función exponencial.

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
        error('Método no válido para calcular la exponencial');
end

% Escalado
A=escalado('expv',metodo_f,plataforma,A,s);

% Evaluamos la expresión polinómica de forma eficiente
switch metodo_f
    case {'taylor','bernoulli','taylor_bernoulli'}
        fAv = polyvalmv(plataforma,p,A,v,s);
    otherwise
        error('Método de cálculo de la función no válido');
end

end
