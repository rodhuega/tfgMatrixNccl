function [m,s,pA,nProd]=selecciona_ms_expv(metodo_f,metodo_ms,plataforma,A)
% [m,s,pA,nProd]=selecciona_ms_expv(metodo_f,metodo_ms,plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz para calcular la acción de la función exponencial.
%
% Datos de entrada:
% - metodo_f:   Método a emplear para calcular exp(A) (taylor, bernoulli,
%               taylor_bernoulli, splines,...).
% - metodo_ms:  Método a usar para calcular el grado del polinomio (m) y 
%               el valor del escalado (s).  Ejemplos: 'conEstNorma' (con 
%               estimaciones de normas de las potencias matriciales), 
%               'sinEstNorma' (sin estimaciones de normas de las potencias 
%               matriciales).
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
%
% Datos de salida:
% - m:          Orden de la aproximación polinómica a f(A). Coincide con el 
%               grado del polinomio de aproximación a f(A).
% - s:          Valor del escalado de la matriz.
% - pA:         Array de celdas de una única componente con la matriz A.
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
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
                error('Método de selección de m y s no contemplado');
        end       
    otherwise
        error('Método no válido para calcular la exponencial'); 
end

end

