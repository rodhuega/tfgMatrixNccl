function pA=escalado(f,metodo_f,plataforma,pA,s)
% pA=escalado(f,metodo_f,plataforma,pA,s)
%
% Escalado de la matriz A y/o de sus potencias.
%
% Datos de entrada:
% - f:          Función a aplicar sobre la matriz ('exp','cos','cosh', ...)
%               o acción de la función ('expv','cosv','coshv', ...).
% - metodo_f:   Método a emplear para calcular f(A) ('taylor', 'bernoulli',
%               'hermite', ...).
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - pA:         Vector de cells arrays con las potencias de A, de modo que 
%               pA{i} contiene:
%               a) A^i, para i=1,2,3,...,q, en el caso de la función 
%                  exponencial o en el caso del coseno por Bernoulli al
%                  emplear todos los términos del polinomio.
%               b) B^i, para i=1,2,3,...,q, siendo B=A*A, en el caso de la 
%                  función coseno o coseno hiperbólico.
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - pA:         Vector de cells arrays con las potencias de A escaladas.

switch f
    % Función matricial
    case 'exp'
        pA=escalado_exp(plataforma,pA,s);
    case {'cos','cosh'}
        switch metodo_f
            case 'bernoulli' 
                formulacion=get_formulacion_cos_bernoulli;
                switch formulacion
                     % Al trabajar con todos los términos, el escalado es 
                     % de tipo exponencial
                    case {'terminos_pares_polinomio_completo','terminos_pares_impares_polinomio_completo'}
                        pA=escalado_exp(plataforma,pA,s);                        
                    case 'terminos_pares_polinomio_solo_pares'
                        pA=escalado_cos(plataforma,pA,s);
                end
            otherwise
                pA=escalado_cos(plataforma,pA,s);
        end
    % Acción de la función matricial (f(A)*v)
    case 'expv'
        pA=escalado_expv(plataforma,pA,s);
    otherwise
        error('Función matricial o acción no contemplada');
end
end



