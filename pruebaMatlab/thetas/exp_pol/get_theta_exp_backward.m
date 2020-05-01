function [theta,M]=get_theta_exp_backward(metodo_f,tipo_error)
%[theta,M]=get_theta_exp_backward(metodo_f,tipo_error)
%
% Proporciona los valores theta de la función exponencial ante errores 
% absolutos o relativos de tipo backward.
% 
% Datos de entrada:
% - metodo_f:   Método a emplear para calcular la función matricial
%               ('taylor', 'bernoulli', 'hermite', ...).
% - tipo_error: Tipo de error ('absoluto' o 'relativo').
%
% Datos de salida:
% - theta:      Vector con los valores theta de la función f para los 
%               grados del polinomio de aproximación recogido en M.
% - M:          Vector con los grados del polinomio.

switch tipo_error
    case 'absoluto'
        switch metodo_f
            case {'taylor','bernoulli','taylor_bernoulli','hermite'}
                % Resultados idénticos para diferentes valores de nt                
                % Theta values of absolute backward errors for exp (nt=250,nd=15)                
                theta=[0.0000087334575100	% m= 2 (nt=250, nd=15)
                       0.0016780188443200	% m= 4 (nt=250, nd=15)
                       0.0177308219965400	% m= 6 (nt=250, nd=15)
                       0.1715477373125700	% m= 9 (nt=250, nd=15)
                       0.3280542018037199	% m=12 (nt=250, nd=15)
                       0.7912740176600197	% m=16 (nt=250, nd=15)
                       1.4150704475615306	% m=20 (nt=250, nd=15)
                       2.5572459340018798	% m=25 (nt=250, nd=15)
                       3.4118771725567640	% m=30 (nt=250, nd=15)
                       4.7855459552778310	% m=36 (nt=250, nd=15)
                       6.2345518738859909	% m=42 (nt=250, nd=15)
                       8.2429594350869753	% m=49 (nt=250, nd=15)
                       9.7882040407606574	% m=56 (nt=250, nd=15)
                      11.8840247957303529]';% m=64 (nt=250, nd=15)  
                M=[2 4 6 9 12 16 20 25 30 36 42 49 56 64];                
            otherwise
                error('Método no válido para calcular la exponencial'); 
        end
    case 'relativo'
        switch metodo_f
            case {'taylor','bernoulli','taylor_bernoulli','hermite'}
                % Resultados idénticos para diferentes valores de nt                   
                % Theta values of relative backward errors for exp (nt=250,nd=15)
                theta=[0.0000000258095600	 % m= 2 (nt=250, nd=15)
                       0.0003397168839900	 % m= 4 (nt=250, nd=15)
                       0.0090656564075900	 % m= 6 (nt=250, nd=15)
                       0.1441829761614301	 % m= 9 (nt=250, nd=15)
                       0.2996158913811499	 % m=12 (nt=250, nd=15)
                       0.7802874256626497	 % m=16 (nt=250, nd=15)
                       1.4382525968043300	 % m=20 (nt=250, nd=15)
                       2.6428534574594327	 % m=25 (nt=250, nd=15)
                       3.5396663487436828	 % m=30 (nt=250, nd=15)
                       4.9729156261919778	 % m=36 (nt=250, nd=15)
                       6.4756827360799791	 % m=42 (nt=250, nd=15)                       
                       8.5469020456849254	 % m=49 (nt=250, nd=15)
                      10.1334283178974758	 % m=56 (nt=250, nd=15)
                      12.2778370232468905]'; % m=64 (nt=250, nd=15)
                M=[2 4 6 9 12 16 20 25 30 36 42 49 56 64]; 
            otherwise
                error('Método no válido para calcular la exponencial');         
        end
    otherwise
         error('Tipo de error no contemplado');
end
end        


