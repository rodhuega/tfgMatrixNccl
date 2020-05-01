function fAv=expv_pol(metodo_f,plataforma,A,v,m,s)
% fAv=expv_pol(metodo_f,plataforma,A,v,m,s)
%
% Funcion principal para el c�lculo de la acci�n de la exponencial de una 
% matriz.
%
% Datos de entrada:
% - metodo_f:   M�todo a emplear para calcular la exponencial (taylor, 
%               bernoulli, hermite, ...).
% - plataforma: Decide si calculamos la acci�n matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz a calcular su exponencial expresada en forma de un
%               array de celdas de una �nica componente.
% - v:          Vector columna para calcular el valor de la acci�n de la 
%               funci�n matricial, es decir, f(A)*v sin obtener f(A) de 
%               forma expl�cita. 
% - m:          Orden de la aproximaci�n polin�mica a f(A). 
% - s:          Valor del escalado de la matriz A.
%
% Datos de salida:
% - fA:         Valor de la acci�n de la funci�n exponencial.

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
A=escalado('expv',metodo_f,plataforma,A,s);

% Evaluamos la expresi�n polin�mica de forma eficiente
switch metodo_f
    case {'taylor','bernoulli','taylor_bernoulli'}
        fAv = polyvalmv(plataforma,p,A,v,s);
    otherwise
        error('M�todo de c�lculo de la funci�n no v�lido');
end

end
