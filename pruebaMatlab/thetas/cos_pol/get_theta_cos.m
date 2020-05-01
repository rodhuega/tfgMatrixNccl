function [theta,M]=get_theta_cos(metodo_f,tipo_error_1,tipo_error_2)
%[theta,M]=get_theta_cos(metodo_f,tipo_error_1,tipo_error_2)
%
% Proporciona los valores theta de la función coseno, ante errores de tipo 
% forward o backward, absolutos o relativos.
% 
% Datos de entrada:
% - metodo_f:     Método a emplear para calcular la función matricial
%                 ('taylor', 'bernoulli', 'hermite', ...).
% - tipo_error_1: Tipo de error ('forward' o 'backward').
% - tipo_error_2: Tipo de error ('absoluto' o 'relativo').
%
% Datos de salida:
% - theta:        Vector con los valores theta de la función f para los 
%                 grados del polinomio de aproximación recogido en M.
% - M:            Vector con los grados del polinomio.

switch tipo_error_1
    case 'forward'
        [theta,M]=get_theta_cos_forward(metodo_f,tipo_error_2);
    case 'backward'
        [theta,M]=get_theta_cos_backward(metodo_f,tipo_error_2); 
    otherwise
         error('Tipo de error no contemplado');
end
end
