function [m,s,pA,nProd]=selecciona_ms_exp(metodo_f,metodo_ms,plataforma,A)
% [m,s,pA,nProd]=selecciona_ms_exp(metodo_f,metodo_ms,plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz para aplicar la función exponencial sobre A.
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
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i} 
%               contiene A^i, para i=1,2,3,...,q, siendo q=ceil(sqrt(m)) o 
%               q=floor(sqrt(m)).
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.


switch metodo_f
    case {'taylor','taylor_bernoulli'}
        mmin=2;
        mmax=30;
        switch metodo_ms
            case 'conEstNorma'
                %mmax=30;
                [m,s,pA,nProd]=selecciona_ms_exp_taylor_conEstNorma(plataforma,A,mmax);
            case 'sinEstNormaSplines'        
                [m,s,pA,nProd]=selecciona_ms_exp_splines_sinEstNorma(plataforma,A);
            case 'conEstNormaSplines'
                [m,s,pA,nProd]=selecciona_ms_exp_splines_conEstNorma(plataforma,A);                
            case 'conEstNormaSinPotencias'
                %mmax=64;            
                [m,s,pA,nProd]=selecciona_ms_conEstNormaSinPotencias('exp',metodo_f,plataforma,A,mmin,mmax);
            case 'conEstNormaConPotencias'
                %mmax=64;              
                [m,s,pA,nProd]=selecciona_ms_conEstNormaConPotencias('exp',metodo_f,plataforma,A,mmin,mmax);
            case 'conEstNormaConPotenciasNuevo'
                %mmax=64;               
                [m,s,pA,nProd]=selecciona_ms_conEstNormaConPotenciasNuevo('exp',metodo_f,plataforma,A,mmin,mmax);                
            otherwise
                error('Método de selección de m y s no contemplado');
        end
    case 'bernoulli'
        mmin=2;
        mmax=30;
        switch metodo_ms
            case 'conEstNorma'
                %mmax=30;
                [m,s,pA,nProd]=selecciona_ms_exp_taylor_conEstNorma(plataforma,A,mmax);
            case 'sinEstNormaSplines'        
                [m,s,pA,nProd]=selecciona_ms_exp_splines_sinEstNorma(plataforma,A);
            case 'conEstNormaSplines'
                [m,s,pA,nProd]=selecciona_ms_exp_splines_conEstNorma(plataforma,A);                
            case 'conEstNormaSinPotencias'
                %mmax=64;            
                [m,s,pA,nProd]=selecciona_ms_conEstNormaSinPotencias('exp',metodo_f,plataforma,A,mmin,mmax);
            case 'conEstNormaConPotencias'
                %mmax=64;              
                [m,s,pA,nProd]=selecciona_ms_conEstNormaConPotencias('exp',metodo_f,plataforma,A,mmin,mmax);
            case 'conEstNormaConPotenciasNuevo'
                %mmax=64;                            
                [m,s,pA,nProd]=selecciona_ms_conEstNormaConPotenciasNuevo('exp',metodo_f,plataforma,A,mmin,mmax);                
            otherwise
                error('Método de selección de m y s no contemplado');
        end        
    case 'taylor_sastre'
        mmax=30;
        switch metodo_ms
            case 'sinEstNorma'
                [m,s,pA,nProd]=selecciona_ms_exp_taylor_sastre_sinEstNorma(plataforma,A,mmax);            
            case 'conEstNorma'
                [m,s,pA,nProd]=selecciona_ms_exp_taylor_sastre_conEstNorma(plataforma,A,mmax);
            otherwise
                error('Método de selección de m y s no contemplado');
        end                
    case 'splines'
        switch metodo_ms
            case 'sinEstNormaSplines'        
                [m,s,pA,nProd]=selecciona_ms_exp_splines_sinEstNorma(plataforma,A);
            case 'conEstNormaSplines'
                [m,s,pA,nProd]=selecciona_ms_exp_splines_conEstNorma(plataforma,A);
            otherwise
                error('Método de selección de m y s no contemplado');
        end        
    otherwise
        error('Método no válido para calcular la exponencial'); 
end

end

