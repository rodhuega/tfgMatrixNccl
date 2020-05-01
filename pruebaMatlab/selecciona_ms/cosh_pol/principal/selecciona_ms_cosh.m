function [m,s,pB,nProd]=selecciona_ms_cosh(metodo_f,metodo_ms,plataforma,A)
% [m,s,pA,nProd]=selecciona_ms_cosh(metodo_f,metodo_ms,plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz para aplicar la función coseno hiperbólico sobre A.
%
% Datos de entrada:
% - metodo_f:   Método a emplear para calcular cosh(A) (taylor, hermite,
%               ...).
% - metodo_ms:  Método a usar para calcular el grado del polinomio (m) y 
%               el valor del escalado (s).
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:           Matriz de la cual calculamos f(A).
%
% Datos de salida:
% - m:          Orden de la aproximación polinómica a f(A). El grado del 
%               polinomio de aproximación a f(A) será 2*m.
% - s:          Valor del escalado de la matriz.
% - pB:         Vector de celdas con las potencias de B=A^2, de modo que
%               pB{i} contiene B^i, para i=1,2,3,...,q, siendo q<=4.
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

switch metodo_f
    %case 'taylor'
    %    switch metodo_ms
    %        case 'sinEstNorma'
    %            [m,s,pB,nProd]=selecciona_ms_cosh_taylor_sinEstNorma(plataforma,A);
    %        case 'conEstNorma'
    %            [m,s,pB,nProd]=selecciona_ms_cosh_taylor_conEstNorma(plataforma,A);
    %        otherwise
    %            error('Método de selección de m y s no contemplado');
    %    end 
    case {'bernoulli'}
        mmin=2;
        mmax=49;
        switch metodo_ms
            case 'conEstNormaSinPotencias'
                [m,s,pB,nProd]=selecciona_ms_conEstNormaSinPotencias('cosh',metodo_f,plataforma,A,mmin,mmax);
            case 'conEstNormaConPotencias'
                [m,s,pB,nProd]=selecciona_ms_conEstNormaConPotencias('cosh',metodo_f,plataforma,A,mmin,mmax);
            case 'conEstNormaConPotenciasNuevo'
                [m,s,pB,nProd]=selecciona_ms_conEstNormaConPotenciasNuevo('cosh',metodo_f,plataforma,A,mmin,mmax);                
            otherwise
                error('Método de selección de m y s no contemplado');
        end    
    case 'hermite'
        switch metodo_ms
            case 'sinEstNorma'
                [m,s,pB,nProd]=selecciona_ms_cosh_hermite_sinEstNorma(plataforma,A);
            case 'conEstNorma'
                [m,s,pB,nProd]=selecciona_ms_cosh_hermite_conEstNorma(plataforma,A);                
            otherwise
                error('Método de selección de m y s no contemplado');
        end        
    otherwise
        error('Método no válido para calcular el coseno hiperbólico'); 
end

end


