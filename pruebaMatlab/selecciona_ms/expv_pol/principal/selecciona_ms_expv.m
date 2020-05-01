function [m,s,pA,nProd]=selecciona_ms_expv(metodo_f,metodo_ms,plataforma,A)
% [m,s,pA,nProd]=selecciona_ms_expv(metodo_f,metodo_ms,plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz para calcular la acci�n de la funci�n exponencial.
%
% Datos de entrada:
% - metodo_f:   M�todo a emplear para calcular exp(A) (taylor, bernoulli,
%               taylor_bernoulli, splines,...).
% - metodo_ms:  M�todo a usar para calcular el grado del polinomio (m) y 
%               el valor del escalado (s).  Ejemplos: 'conEstNorma' (con 
%               estimaciones de normas de las potencias matriciales), 
%               'sinEstNorma' (sin estimaciones de normas de las potencias 
%               matriciales).
% - plataforma: Decide si calculamos la funci�n matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
%
% Datos de salida:
% - m:          Orden de la aproximaci�n polin�mica a f(A). Coincide con el 
%               grado del polinomio de aproximaci�n a f(A).
% - s:          Valor del escalado de la matriz.
% - pA:         Array de celdas de una �nica componente con la matriz A.
% - nProd:      N�mero de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

switch metodo_f
    case {'taylor','bernoulli','taylor_bernoulli'}
        switch metodo_ms
            case 'conEstNorma'
                mmax=30;
                [m,s,pA,nProd]=selecciona_ms_expv_taylor_conEstNorma(plataforma,A,mmax);
            case 'conEstNormaSinPotencias'
                mmin=2;
                mmax=64;
                %[m,s,pA,nProd]=selecciona_ms_expv_taylor_conEstNormaSolo(plataforma,A,mmax);
                [m,s,pA,nProd]=selecciona_ms_conEstNormaSinPotencias('expv',metodo_f,plataforma,A,mmin,mmax);
            case 'conEstNormaConPotencias'
                mmin=2;
                mmax=64;                
                %[m,s,pA,nProd]=selecciona_ms_expv_taylor_conEstNormaPotencias(plataforma,A,mmax);                
                [m,s,pA,nProd]=selecciona_ms_conEstNormaConPotencias('expv',metodo_f,plataforma,A,mmin,mmax);
            otherwise
                error('M�todo de selecci�n de m y s no contemplado');
        end       
    otherwise
        error('M�todo no v�lido para calcular la exponencial'); 
end

end

