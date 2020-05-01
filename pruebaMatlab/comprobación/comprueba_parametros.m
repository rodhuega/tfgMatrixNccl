function comprueba_parametros(f,metodo_f,metodo_ms,plataforma)
% comprueba_parametros(f,metodo_f,metodo_ms,plataforma)
%
% Función encargada de comprobar que los parámetros de entrada empleados 
% correctos. En caso de que no lo sean, se muestra un mensaje de error.
%
% Datos de entrada:
% - f:          Función a aplicar sobre la matriz ('exp','cos','cosh', ...)
%               o acción de la función ('expv','cosv','coshv', ...).
% - metodo_f:   Método a emplear para calcular f(A) ('taylor', 'bernoulli',
%               'hermite', ...).
% - metodo_ms:  Método a usar para calcular el grado del polinomio (m) y 
%               el valor del escalado (s). Ejemplos: 'conEstNorma' (con 
%               estimaciones de normas de las potencias matriciales), 
%               'sinEstNorma' (sin estimaciones de normas de las potencias 
%               matriciales).
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').

switch f % Función matricial
    case 'exp'        
        switch metodo_f % Método de cálculo
            case {'taylor','bernoulli','taylor_bernoulli'}                
                switch metodo_ms % Método para obtener m y s
                    case {'conEstNorma','sinEstNormaSplines','conEstNormaSplines','conEstNormaSinPotencias','conEstNormaConPotencias','conEstNormaConPotenciasNuevo'}                       
                        switch plataforma  % Plataforma
                            case {'sinGPUs','conGPUs'}
                            otherwise
                                error('Plataforma destino incorrecta');
                        end
                    otherwise
                        error('Método de selección de m y s no contemplado');
                end
            case 'taylor_sastre'                
                switch metodo_ms % Método para obtener m y s
                    case {'sinEstNorma','conEstNorma'}                       
                        switch plataforma  % Plataforma
                            case 'sinGPUs'
                            case 'conGPUs'
                                error('Plataforma destino incorrecta');
                            otherwise
                                error('Plataforma destino incorrecta');
                        end
                    otherwise
                        error('Método de selección de m y s no contemplado');
                end
            case 'splines'                
                switch metodo_ms % Método para obtener m y s
                    case {'sinEstNormaSplines','conEstNormaSplines'}                       
                        switch plataforma  % Plataforma
                            case 'sinGPUs'
                            case 'conGPUs'
                                error('Plataforma destino incorrecta');
                            otherwise
                                error('Plataforma destino incorrecta');
                        end
                    otherwise
                        error('Método de selección de m y s no contemplado');
                end                 
            otherwise
                error('Método no válido para calcular la exponencial'); 
        end
    case 'cos'
        switch metodo_f % Método de cálculo
            case 'taylor'
                switch metodo_ms % Método para obtener m y s
                    case {'sinEstNorma','conEstNorma','conEstNormaSinPotencias','conEstNormaConPotencias','conEstNormaConPotenciasNuevo'}                       
                        switch plataforma  % Plataforma
                            case {'sinGPUs','conGPUs'}
                            otherwise
                                error('Plataforma destino incorrecta');
                        end
                    otherwise
                        error('Método de selección de m y s no contemplado');
                end        
             case 'bernoulli'
                formulacion=get_formulacion_cos_bernoulli;
                if isempty(formulacion)==1
                    switch metodo_ms % Método para obtener m y s
                        case {'sinEstNorma','conEstNorma','conEstNormaSinPotencias','conEstNormaConPotencias','conEstNormaConPotenciasNuevo'}                                              
                            switch plataforma  % Plataforma
                                case {'sinGPUs','conGPUs'}
                                otherwise
                                    error('Plataforma destino incorrecta');
                            end
                        otherwise
                            error('Método de selección de m y s no contemplado');
                    end
                else
                    switch formulacion
                        case {'terminos_pares_polinomio_completo','terminos_pares_impares_polinomio_completo'}
                            switch metodo_ms % Método para obtener m y s
                                case {'conEstNormaSinPotencias','conEstNormaConPotencias','conEstNormaConPotenciasNuevo'}                                              
                                    switch plataforma  % Plataforma
                                        case {'sinGPUs','conGPUs'}
                                        otherwise
                                        error('Plataforma destino incorrecta');
                                    end
                                otherwise
                                    error('Método de selección de m y s no contemplado');
                            end
                        case {'terminos_pares_polinomio_solo_pares'}
                            switch metodo_ms % Método para obtener m y s
                                case {'sinEstNorma','conEstNorma','conEstNormaSinPotencias','conEstNormaConPotencias','conEstNormaConPotenciasNuevo'}                       
                                    switch plataforma  % Plataforma
                                        case {'sinGPUs','conGPUs'}
                                        otherwise
                                            error('Plataforma destino incorrecta');
                                    end
                                otherwise
                                    error('Método de selección de m y s no contemplado');
                            end
                        otherwise
                            error('Formulación teórica del método de Bernoulli no contemplada');
                    end
                end
             case 'hermite'
                switch metodo_ms % Método para obtener m y s
                    case {'sinEstNorma'}                       
                        switch plataforma  % Plataforma
                            case {'sinGPUs','conGPUs'}
                            otherwise
                                error('Plataforma destino incorrecta');
                        end
                    otherwise
                        error('Método de selección de m y s no contemplado');
                end 
             case 'taylor_sastre'
                switch metodo_ms % Método para obtener m y s
                    case {'sinEstNorma','conEstNorma'}                       
                        switch plataforma  % Plataforma
                            case 'sinGPUs'
                            case 'conGPUs'
                                error('Plataforma destino incorrecta');
                            otherwise
                                error('Plataforma destino incorrecta');
                        end
                    otherwise
                        error('Método de selección de m y s no contemplado');
                end                          
            otherwise
                error('Método no válido para calcular el coseno');
        end
    case 'cosh'
        switch metodo_f % Método de cálculo
            case 'bernoulli'
                switch metodo_ms % Método para obtener m y s
                    case {'conEstNormaSinPotencias','conEstNormaConPotencias','conEstNormaConPotenciasNuevo'}                       
                        switch plataforma  % Plataforma
                            case {'sinGPUs','conGPUs'}
                            otherwise
                                error('Plataforma destino incorrecta');
                        end
                    otherwise
                        error('Método de selección de m y s no contemplado');
                end         
             case 'hermite'
                switch metodo_ms % Método para obtener m y s
                    case {'sinEstNorma','conEstNorma'}                       
                        switch plataforma  % Plataforma
                            case {'sinGPUs','conGPUs'}
                            otherwise
                                error('Plataforma destino incorrecta');
                        end
                    otherwise
                        error('Método de selección de m y s no contemplado');
                end                                
            otherwise
                error('Método no válido para calcular el coseno hiperbólico');
        end        
    % Acción de la función matricial (f(A)*v)
     case 'expv'
        switch metodo_f % Método de cálculo
            case {'taylor','bernoulli','taylor_bernoulli'}                
                switch metodo_ms % Método para obtener m y s
                    case {'conEstNorma','conEstNormaSinPotencias','conEstNormaConPotencias','conEstNormaConPotenciasNuevo'}                       
                        switch plataforma  % Plataforma
                            case 'sinGPUs'
                            case 'conGPUs'
                                error('Plataforma destino incorrecta');
                            otherwise
                                error('Plataforma destino incorrecta');
                        end
                    otherwise
                        error('Método de selección de m y s no contemplado');
                end
            otherwise
                error('Método no válido para calcular la acción exponencial'); 
        end   
    otherwise
        error('Función matricial o acción no contemplada');
end
end  

