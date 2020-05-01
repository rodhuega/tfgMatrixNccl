function [theta,M]=get_theta(f,metodo_f,tipo_error_1,tipo_error_2)
%[theta,M]=get_theta(f,metodo_f,tipo_error_1,tipo_error_2)
%
% Proporciona los valores theta de la función f, ante errores de tipo 
% forward o backward, absolutos o relativos.
% 
% Datos de entrada:
% - f:            Función a aplicar sobre la matriz ('exp','cos','cosh').
% - metodo_f:     Método a emplear para calcular la función matricial
%                 ('taylor', 'bernoulli', 'hermite', ...).
% - tipo_error_1: Tipo de error ('forward' o 'backward').
% - tipo_error_2: Tipo de error ('absoluto' o 'relativo').
%
% Datos de salida:
% - theta:        Vector con los valores theta de la función f para los 
%                 grados del polinomio de aproximación recogido en M.
% - M:            Vector con los grados del polinomio.

switch f
    % Función matricial
    case 'exp'
        [theta,M]=get_theta_exp(metodo_f,tipo_error_1,tipo_error_2);
    case {'cos','cosh'} % Los valores coinciden para cos y cosh
        [theta,M]=get_theta_cos(metodo_f,tipo_error_1,tipo_error_2);
    otherwise
        error('Función matricial no contemplada');
end
end