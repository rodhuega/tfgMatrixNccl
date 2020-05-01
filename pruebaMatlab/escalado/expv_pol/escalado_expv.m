function A = escalado_expv(plataforma,A,s)
% A=escalado_expv(plataforma,A,s)
%
% Escalado de la matriz A.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz a escalar expresada en forma de un array de celdas 
%               de una única componente.
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - A:          Vector de cells arrays de una única componente con la 
%               matriz A escalada.

switch plataforma
    case 'sinGPUs'
        A = escalado_expv_sinGPUs(A,s);
    case 'conGPUs'
        escalado_expv_conGPUs(s);
    otherwise
        error('Plataforma destino incorrecta');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = escalado_expv_sinGPUs(A,s)
% A=escalado_expv_sinGPUs(plataforma,A,s)
%
% Escalado de la matriz A sin emplear GPUs.
%
% Datos de entrada:
% - A:  Matriz a escalar expresada en forma de un array de celdas de una 
%       única componente.
% - s:  Valor del escalado de la matriz.
%
% Datos de salida:
% - A:  Vector de cells arrays de una única componente con la matriz A 
%       escalada.

if s>1
    A{1}=A{1}/s;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function escalado_expv_conGPUs(s)
% escalado_expv_conGPUs(s)
%
% Escalado de la matriz A mediante GPUs.
%
% Datos de entrada:
% - s:  Valor del escalado de la matriz.

call_gpu('scale',s);

end



