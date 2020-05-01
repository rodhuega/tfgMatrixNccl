function inicializa_plataforma(f,metodo_f,plataforma,A)
% inicializa_plataforma(f,metodo_f,plataforma,A)
%
% Funci�n encargada de inicializar la plataforma de c�lculo y determinadas
% variables globales.
%
% Datos de entrada:
% - f:          Funci�n a aplicar sobre la matriz ('exp','cos','cosh', ...)
%               o acci�n de la funci�n ('expv','cosv','coshv', ...).
% - metodo_f:   M�todo a emplear para calcular f(A) ('taylor', 'bernoulli',
%               'hermite', ...).
% - plataforma: Decide la plataforma destino en la cual calculamos la
%               funci�n matricial. Tomar� como valores 'sinGPUs' (si 
%               empleamos Matlab) o 'conGPUs' (si usamos las GPUs).
% - A:          Matriz de la cual calculamos f(A).

% Inicializamos variables globales

switch f
    case 'cos'
        switch metodo_f
            case 'bernoulli'
                formulacion=get_formulacion_cos_bernoulli;
                if isempty(formulacion)
                    %formulacion='terminos_pares_polinomio_completo';
                    formulacion='terminos_pares_impares_polinomio_completo';
                    %formulacion='terminos_pares_polinomio_solo_pares';
                    set_formulacion_cos_bernoulli(formulacion);
                end
        end
end

% Inicializamos la plataforma de c�lculo

switch plataforma
    case 'sinGPUs'
    case 'conGPUs'
        inicializa_plataforma_gpu(f,metodo_f,A);
    otherwise
        error('Plataforma destino incorrecta');
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function inicializa_plataforma_gpu(f,metodo_f,A)
% inicializa_plataforma_gpu(f,metodo_f,A)
%
% Funci�n encargada de inicializar la plataforma de c�lculo basada en GPUs.
%
% Datos de entrada:
% - f:        Funci�n a aplicar sobre la matriz ('exp','cos','cosh', ...)
%             o acci�n de la funci�n ('expv','cosv','coshv', ...).
% - metodo_f: M�todo a emplear para calcular f(A) ('taylor', 'bernoulli',
%             'hermite', ...).
% - A:        Matriz de la cual calculamos f(A).

call_gpu('init',f,metodo_f,A);
end
