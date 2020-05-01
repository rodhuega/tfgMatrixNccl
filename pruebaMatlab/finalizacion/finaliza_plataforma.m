function fA=finaliza_plataforma(plataforma,fA)
% fA=finaliza_plataforma(plataforma,fA)
%
% Función encargada de finalizar la plataforma de cálculo.
%
% Datos de entrada:
% - plataforma: Decide la plataforma destino en la cual calculamos la
%               función matricial. Tomará como valores 'sinGPUs' (si 
%               empleamos Matlab) o 'conGPUs' (si usamos las GPUs).
% - fA:         Valor de la función matricial sobre la matriz. Sólo se
%               tendrá en cuenta con la opción 'sinGPUs'.
%
% Datos de salida:
% - fA:         Valor de la función matricial sobre la matriz. En caso de 
%               no emplear GPUs, coincidirá con el dato de entrada. 
%      

switch plataforma
    case 'sinGPUs'
    case 'conGPUs'
        fA=finaliza_plataforma_gpu;
    otherwise
        error('Plataforma destino incorrecta');
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function fA=finaliza_plataforma_gpu
% fA=finaliza_plataforma_gpu
%
% Función encargada de finalizar la plataforma de cálculo basada en GPUs.
%
% Datos de salida:
% - fA: Valor de la función matricial sobre la matriz.

fA = call_gpu('finalize');
end
