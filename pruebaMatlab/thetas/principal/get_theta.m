function [theta,M]=get_theta(f,metodo_f,tipo_error_1,tipo_error_2)
%[theta,M]=get_theta(f,metodo_f,tipo_error_1,tipo_error_2)
%
% Proporciona los valores theta de la funci�n f, ante errores de tipo 
% forward o backward, absolutos o relativos.
% 
% Datos de entrada:
% - f:            Funci�n a aplicar sobre la matriz ('exp','cos','cosh').
% - metodo_f:     M�todo a emplear para calcular la funci�n matricial
%                 ('taylor', 'bernoulli', 'hermite', ...).
% - tipo_error_1: Tipo de error ('forward' o 'backward').
% - tipo_error_2: Tipo de error ('absoluto' o 'relativo').
%
% Datos de salida:
% - theta:        Vector con los valores theta de la funci�n f para los 
%                 grados del polinomio de aproximaci�n recogido en M.
% - M:            Vector con los grados del polinomio.

switch f
    % Funci�n matricial
    case 'exp'
        [theta,M]=get_theta_exp(metodo_f,tipo_error_1,tipo_error_2);
    case {'cos','cosh'} % Los valores coinciden para cos y cosh
        [theta,M]=get_theta_cos(metodo_f,tipo_error_1,tipo_error_2);
    otherwise
        error('Funci�n matricial no contemplada');
end
end