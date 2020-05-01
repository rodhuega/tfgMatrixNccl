function [A,nProd]=escalado_regresivo(f,plataforma,A,s)
% [A,nProd]=escalado_regresivo(f,plataforma,A,s)
%
% Escalado regresivo de la matriz A.
%
% Datos de entrada:
% - f:          Función a aplicar sobre la matriz ('exp','cos','cosh', ...).
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz resultado de la evaluación del polinomio de 
%               aproximación.
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - A:          Resultado tras el escalado regresivo.
% - nProd:      Número de productos matriciales que se han llevado a cabo
%               con el escalado.
switch f
    case 'exp'
        [A,nProd]=escalado_regresivo_exp(plataforma,A,s);
    case {'cos','cosh'}
        [A,nProd]=escalado_regresivo_cos(plataforma,A,s);        
    otherwise
        error('Función matricial no contemplada');
end
end



