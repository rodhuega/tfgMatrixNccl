function pB = escalado_cos(plataforma,pB,s)
% pB=escalado_cos(plataforma,pB,s)
%
% Escalado de las potencias de la matriz B=A^2.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - pB:         Vector de cells arrays con las potencias de B, de modo que
%               pB{i} contiene B^i, para i=1,2,3,...,q, siendo B=A*A.
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - pB:         Vector de cells arrays con las potencias de B=A^2 escaladas.

switch plataforma
    case 'sinGPUs'
        pB = escalado_cos_sinGPUs(pB,s);
    case 'conGPUs'
        escalado_cos_conGPUs(s);
    otherwise
        error('Plataforma destino incorrecta');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pB=escalado_cos_sinGPUs(pB,s)
% pB=escalado_cos_sinGPUs(pB,s)
%
% Escalado de las potencias de la matriz B=A^2 sin emplear GPUs.
%
% Datos de entrada:
% - pB: Vector de cells arrays con las potencias de B, de modo que
%       pB{i} contiene B^i, para i=1,2,3,...,q, siendo B=A*A.
% - s:  Valor del escalado de la matriz.
%
% Datos de salida:
% - pB: Vector de cells arrays con las potencias de B=A^2 escaladas.

if s>0
    q=length(pB);
    for k=1:q
        pB{k}=pB{k}/(4^(k*s));
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function escalado_cos_conGPUs(s)
% escalado_cos_conGPUs(s)
%
% Escalado de las potencias de la matriz B=A^2 mediante GPUs.
%
% Datos de entrada:
% - s: Valor del escalado de la matriz.
%

call_gpu('scale',s);

end

