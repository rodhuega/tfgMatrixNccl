function fA=finaliza_plataforma(plataforma,fA)
% fA=finaliza_plataforma(plataforma,fA)
%
% Funci�n encargada de finalizar la plataforma de c�lculo.
%
% Datos de entrada:
% - plataforma: Decide la plataforma destino en la cual calculamos la
%               funci�n matricial. Tomar� como valores 'sinGPUs' (si 
%               empleamos Matlab) o 'conGPUs' (si usamos las GPUs).
% - fA:         Valor de la funci�n matricial sobre la matriz. S�lo se
%               tendr� en cuenta con la opci�n 'sinGPUs'.
%
% Datos de salida:
% - fA:         Valor de la funci�n matricial sobre la matriz. En caso de 
%               no emplear GPUs, coincidir� con el dato de entrada. 
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
% Funci�n encargada de finalizar la plataforma de c�lculo basada en GPUs.
%
% Datos de salida:
% - fA: Valor de la funci�n matricial sobre la matriz.

fA = call_gpu('finalize');
end
