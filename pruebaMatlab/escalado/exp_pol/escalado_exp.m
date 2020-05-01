function pA = escalado_exp(plataforma,pA,s)
% pA=escalado_exp(plataforma,pA,s)
%
% Escalado de las potencias de la matriz A.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - pA:         Vector de cells arrays con las potencias de A, de modo que
%               pA{i} contiene A^i, para i=1,2,3,...,q.
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - pA:         Vector de cells arrays con las potencias de A escaladas.

switch plataforma
    case 'sinGPUs'
        pA = escalado_exp_sinGPUs(pA,s);
    case 'conGPUs'
        escalado_exp_conGPUs(s);
    otherwise
        error('Plataforma destino incorrecta');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pA = escalado_exp_sinGPUs(pA,s)
% pA=escalado_exp_sinGPUs(plataforma,pA,s)
%
% Escalado de las potencias de la matriz A sin emplear GPUs.
%
% Datos de entrada:
% - pA: Vector de cells arrays con las potencias de A, de modo que pA{i} 
%       contiene A^i, para i=1,2,3,...,q.
% - s:  Valor del escalado de la matriz.
%
% Datos de salida:
% - pA: Vector de cells arrays con las potencias de A escaladas.

if s>0
    q=length(pA);
    for k=1:q
        pA{k}=pA{k}/(2^(k*s));
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function escalado_exp_conGPUs(s)
% escalado_exp_conGPUs(s)
%
% Escalado de las potencias de la matriz A mediante GPUs.
%
% Datos de entrada:
% - s:  Valor del escalado de la matriz.

call_gpu('scale',s);

end



