function pA=escalado(f,metodo_f,plataforma,pA,s)
% pA=escalado(f,metodo_f,plataforma,pA,s)
%
% Escalado de la matriz A y/o de sus potencias.
%
% Datos de entrada:
% - f:          Funci�n a aplicar sobre la matriz ('exp','cos','cosh', ...)
%               o acci�n de la funci�n ('expv','cosv','coshv', ...).
% - metodo_f:   M�todo a emplear para calcular f(A) ('taylor', 'bernoulli',
%               'hermite', ...).
% - plataforma: Decide si calculamos la funci�n matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - pA:         Vector de cells arrays con las potencias de A, de modo que 
%               pA{i} contiene:
%               a) A^i, para i=1,2,3,...,q, en el caso de la funci�n 
%                  exponencial o en el caso del coseno por Bernoulli al
%                  emplear todos los t�rminos del polinomio.
%               b) B^i, para i=1,2,3,...,q, siendo B=A*A, en el caso de la 
%                  funci�n coseno o coseno hiperb�lico.
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - pA:         Vector de cells arrays con las potencias de A escaladas.

switch f
    % Funci�n matricial
    case 'exp'
        pA=escalado_exp(plataforma,pA,s);
    case {'cos','cosh'}
        switch metodo_f
            case 'bernoulli' 
                formulacion=get_formulacion_cos_bernoulli;
                switch formulacion
                     % Al trabajar con todos los t�rminos, el escalado es 
                     % de tipo exponencial
                    case {'terminos_pares_polinomio_completo','terminos_pares_impares_polinomio_completo'}
                        pA=escalado_exp(plataforma,pA,s);                        
                    case 'terminos_pares_polinomio_solo_pares'
                        pA=escalado_cos(plataforma,pA,s);
                end
            otherwise
                pA=escalado_cos(plataforma,pA,s);
        end
    % Acci�n de la funci�n matricial (f(A)*v)
    case 'expv'
        pA=escalado_expv(plataforma,pA,s);
    otherwise
        error('Funci�n matricial o acci�n no contemplada');
end
end



