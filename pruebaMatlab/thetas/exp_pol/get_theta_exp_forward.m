function [theta,M]=get_theta_exp_forward(metodo_f,tipo_error)
%[theta,M]=get_theta_exp_forward(metodo_f,tipo_error)
%
% Proporciona los valores theta de la función exponencial ante errores 
% absolutos o relativos de tipo forward.
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
                % Theta values of absolute forward errors for exp (nt=250,nd=15)
                % Resultados idénticos para diferentes valores de nt
                theta=[0.0000087334702200	% m= 2 (nt=250, nd=15)
                       0.0016783942982700	% m= 4 (nt=250, nd=15)
                       0.0177645270836800	% m= 6 (nt=250, nd=15)
                       0.1148317474773901	% m= 9 (nt=250, nd=15)
                       0.3352136878286099	% m=12 (nt=250, nd=15)
                       0.8246031916386004	% m=16 (nt=250, nd=15)
                       1.5041473223951602	% m=20 (nt=250, nd=15)
                       2.5585766884181314	% m=25 (nt=250, nd=15)
                       3.7810696269831308	% m=30 (nt=250, nd=15)
                       5.4064650937902865	% m=36 (nt=250, nd=15)
                       7.1556200904384877	% m=42 (nt=250, nd=15)                       
                       9.3073843996022045	% m=49 (nt=250, nd=15)
                      11.5453483152121841	% m=56 (nt=250, nd=15)
                      14.1791073371113079]';% m=64 (nt=250, nd=15)
                M=[2 4 6 9 12 16 20 25 30 36 42 49 56 64];                
            otherwise
                error('Método no válido para calcular la exponencial'); 
        end
    case 'relativo'
        switch metodo_f
            case {'taylor','bernoulli','taylor_bernoulli','hermite'}
                % Theta values of relative forward errors for exp (nt=250,nd=15)
                % Resultados idénticos para diferentes valores de nt
                theta=[0.0000087334575100	 % m= 2 (nt=250, nd=15)
                       0.0016780188443200	 % m= 4 (nt=250, nd=15)
                       0.0177308219965400	 % m= 6 (nt=250, nd=15)
                       0.1137689245787801	 % m= 9 (nt=250, nd=15)
                       0.3280542018037199	 % m=12 (nt=250, nd=15)
                       0.7912740176600197	 % m=16 (nt=250, nd=15)
                       1.4150704475615306	 % m=20 (nt=250, nd=15)
                       2.3536427669894207	 % m=25 (nt=250, nd=15)
                       3.4118771725567640	 % m=30 (nt=250, nd=15)
                       4.7855459552778310	 % m=36 (nt=250, nd=15)
                       6.2345518738859909	 % m=42 (nt=250, nd=15)                       
                       7.9882499230847905	 % m=49 (nt=250, nd=15)
                       9.7882040407606574	 % m=56 (nt=250, nd=15)
                      11.8840247957303529]'; % m=64 (nt=250, nd=15)
                M=[2 4 6 9 12 16 20 25 30 36 42 49 56 64]; 
            otherwise
                error('Método no válido para calcular la exponencial');         
        end
    otherwise
         error('Tipo de error no contemplado');
end
end        


