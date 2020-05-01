function [m,s,pB,nProd]=selecciona_ms_cos(metodo_f,metodo_ms,plataforma,A)
% [m,s,pA,nProd]=selecciona_ms_cos(metodo_f,metodo_ms,plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz para aplicar la función coseno sobre A.
%
% Datos de entrada:
% - metodo_f:   Método a emplear para calcular cos(A) (taylor, hermite,
%               ...).
% - metodo_ms:  Método a usar para calcular el grado del polinomio (m) y 
%               el valor del escalado (s).
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
%
% Datos de salida:
% - m:          Orden de la aproximación polinómica a f(A). El grado del 
%               polinomio de aproximación a f(A) será 2*m.
% - s:          Valor del escalado de la matriz.
% - pB:         Vector de celdas con las potencias de B=A^2, de modo que
%               pB{i} contiene B^i, para i=1,2,3,...,q, siendo q<=4.
%               Como excepción, con el método de Bernoulli, y siempre y
%               cuando se trabaje con el polinomio completo, será un vector
%               de celdas con las potencias de A, de modo que pB{i}
%               contendrá A^i, para i=1,2,3,...,q, siendo q=ceil(sqrt(m)).
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

switch metodo_f    
    case 'taylor'
        mmin=2;
        mmax=16;
        switch metodo_ms            
            case 'sinEstNorma'
                [m,s,pB,nProd]=selecciona_ms_cos_taylor_sinEstNorma(plataforma,A);
            case 'conEstNorma'
                [m,s,pB,nProd]=selecciona_ms_cos_taylor_conEstNorma(plataforma,A);
            case 'conEstNormaSinPotencias'
                [m,s,pB,nProd]=selecciona_ms_conEstNormaSinPotencias('cos',metodo_f,plataforma,A,mmin,mmax);
            case 'conEstNormaConPotencias'
                [m,s,pB,nProd]=selecciona_ms_conEstNormaConPotencias('cos',metodo_f,plataforma,A,mmin,mmax);                
            case 'conEstNormaConPotenciasNuevo'
                [m,s,pB,nProd]=selecciona_ms_conEstNormaConPotenciasNuevo('cos',metodo_f,plataforma,A,mmin,mmax);                  
            otherwise
                error('Método de selección de m y s no contemplado');
        end
    case 'bernoulli'
        formulacion=get_formulacion_cos_bernoulli;        
        switch formulacion
            case {'terminos_pares_polinomio_completo','terminos_pares_impares_polinomio_completo'}
                mmin=30;
                mmax=36;
                switch metodo_ms
                    case 'conEstNormaSinPotencias'
                        [m,s,pB,nProd]=selecciona_ms_conEstNormaSinPotencias('cos',metodo_f,plataforma,A,mmin,mmax);
                    case 'conEstNormaConPotencias'
                        [m,s,pB,nProd]=selecciona_ms_conEstNormaConPotencias('cos',metodo_f,plataforma,A,mmin,mmax);
                    case 'conEstNormaConPotenciasNuevo'
                        [m,s,pB,nProd]=selecciona_ms_conEstNormaConPotenciasNuevo('cos',metodo_f,plataforma,A,mmin,mmax);                              
                    otherwise
                        error('Método de selección de m y s no contemplado');
                end
            case 'terminos_pares_polinomio_solo_pares'
                mmin=16;
                mmax=20;
                switch metodo_ms
                    case 'sinEstNorma'
                        [m,s,pB,nProd]=selecciona_ms_cos_taylor_sinEstNorma(plataforma,A);
                    case 'conEstNorma'
                        [m,s,pB,nProd]=selecciona_ms_cos_taylor_conEstNorma(plataforma,A);                    
                    case 'conEstNormaSinPotencias'
                        [m,s,pB,nProd]=selecciona_ms_conEstNormaSinPotencias('cos',metodo_f,plataforma,A,mmin,mmax);                  
                    case 'conEstNormaConPotencias'
                        [m,s,pB,nProd]=selecciona_ms_conEstNormaConPotencias('cos',metodo_f,plataforma,A,mmin,mmax);                 
                    case 'conEstNormaConPotenciasNuevo'
                        [m,s,pB,nProd]=selecciona_ms_conEstNormaConPotenciasNuevo('cos',metodo_f,plataforma,A,mmin,mmax);                           
                    otherwise
                        error('Método de selección de m y s no contemplado');
                end
            otherwise
                error('Formulación de Bernoulli no contemplada');                
        end
    case 'taylor_sastre'
        switch metodo_ms
            case 'sinEstNorma'               
                [m,s,pB,nProd]=selecciona_ms_cos_taylor_sastre_sinEstNorma(plataforma,A);            
            case 'conEstNorma'               
                [m,s,pB,nProd]=selecciona_ms_cos_taylor_sastre_conEstNorma(plataforma,A);
            otherwise
                error('Método de selección de m y s no contemplado');
        end         
    case 'hermite'
        switch metodo_ms
            case 'sinEstNorma'
                [m,s,pB,nProd]=selecciona_ms_cos_hermite_sinEstNorma(plataforma,A);
            otherwise
                error('Método de selección de m y s no contemplado');
        end        
    otherwise
        error('Método no válido para calcular el coseno'); 
end

end


