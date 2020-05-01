function [theta,M]=get_theta_cos_forward(metodo_f,tipo_error)
%[theta,M]=get_theta_cos_forward(metodo_f,tipo_error)
%
% Proporciona los valores theta de la función coseno o coseno hiperbólico
% ante errores absolutos o relativos de tipo forward.
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

% Si el método que escogemos es el de Bernoulli donde sólo empleamos los
% términos pares del polinomio, los valores de theta coincidirán con los de
% Taylor o Hermites.

switch metodo_f
    case 'bernoulli'
        formulacion=get_formulacion_cos_bernoulli;
        switch formulacion
            case 'terminos_pares_polinomio_solo_pares'
                metodo_f='taylor';
        end
end

switch tipo_error
    case 'absoluto'
        switch metodo_f
            case {'taylor','hermite'}
                % Theta values of absolute forward errors for @(x)cos(sqrt(x)) (nt=250, nd=13)
                % Resultados idénticos para diferentes valores de nt
                theta=[0.0000430771990000	% m= 2 (nt=250, nd=13)
                       0.0132137460920000	% m= 4 (nt=250, nd=13)
                       0.1921492462990002	% m= 6 (nt=250, nd=13)
                       1.7498015129630002	% m= 9 (nt=250, nd=13)
                       6.5920076891020001	% m=12 (nt=250, nd=13)
                      21.0870186062700391	% m=16 (nt=250, nd=13)
                      47.3520019672589854	% m=20 (nt=250, nd=13)
                      99.4413296329749699	% m=25 (nt=250, nd=13)
                     174.8690782129047818	% m=30 (nt=250, nd=13)
                     297.9204830753339479	% m=36 (nt=250, nd=13)
                     457.6519665452189543	% m=42 (nt=250, nd=13)
                     691.3637319746217145	% m=49 (nt=250, nd=13)
                     976.7604294039363140	% m=56 (nt=250, nd=13)
                    1366.7813478651726200]';% m=64 (nt=250, nd=13)                 
                M=[2 4 6 9 12 16 20 25 30 36 42 49 56 64];   
            case 'bernoulli' % Polinomio completo
                % Theta values of absolute forward errors for cos or cosh (nt=250, nd=15)
                % Resultados idénticos para diferentes valores de nt
                theta=[0.0002271984518300	% m= 2 (nt=250, nd=15)
                       0.0065633223103200	% m= 4 (nt=250, nd=15)
                       0.0381386632247601	% m= 6 (nt=250, nd=15)
                       0.1149510595534400	% m= 9 (nt=250, nd=15)
                       0.4383483161819298	% m=12 (nt=250, nd=15)
                       0.9810763244656998	% m=16 (nt=250, nd=15)
                       1.7042776030289306	% m=20 (nt=250, nd=15)
                       2.5674905431377990	% m=25 (nt=250, nd=15)
                       4.0560126128455929	% m=30 (nt=250, nd=15)
                       5.7109000664700957	% m=36 (nt=250, nd=15)
                       7.4825284953464193	% m=42 (nt=250, nd=15)                       
                       9.3385619211370834	% m=49 (nt=250, nd=15)
                      11.9081054947739364	% m=56 (nt=250, nd=15)
                      14.5559420698812509]';% m=64 (nt=250, nd=15)
                M=[2 4 6 9 12 16 20 25 30 36 42 49 56 64];                
            otherwise
                error('Método no válido para calcular el coseno'); 
        end
    case 'relativo'
        switch metodo_f
            case {'taylor','hermite'}      
                % Resultados distintos para diferentes valores de nt
                nt=250;
                if nt==200
                    % Theta values of relative forward errors for @(x)cos(sqrt(x)) (nt=200, nd=15)
                    theta=[0.0000430769125600	 % m= 2 (nt=200, nd=15)
                           0.0131968092989200	 % m= 4 (nt=200, nd=15)
                           0.1895232414039101	 % m= 6 (nt=200, nd=15)
                           1.5605489459377009	 % m= 9 (nt=200, nd=15)
                           2.5793699514856598	 % m=12 (nt=200, nd=15)
                           2.8924808231390915	 % m=16 (nt=200, nd=15)
                           3.2469565858100111	 % m=20 (nt=200, nd=15)
                           3.7608633676173122	 % m=25 (nt=200, nd=15)
                           4.3621977559056395	 % m=30 (nt=200, nd=15)
                           5.2132448647262608	 % m=36 (nt=200, nd=15)
                           6.2237944478118585	 % m=42 (nt=200, nd=15)
                           7.6333977365531664	 % m=49 (nt=200, nd=15)
                           9.3282051203741840	 % m=56 (nt=200, nd=15)
                          11.6697045210356549]'; % m=64 (nt=200, nd=15)          
                elseif nt==250
                    % Theta values of relative forward errors for @(x)cos(sqrt(x)) (nt=250, nd=15)                   
                    theta=[0.0000430769125600	 % m= 2 (nt=250, nd=15)
                           0.0131968092989200	 % m= 4 (nt=250, nd=15)
                           0.1895232414039101	 % m= 6 (nt=250, nd=15)
                           1.5605489459377009	 % m= 9 (nt=250, nd=15)
                           2.5553520055399619	 % m=12 (nt=250, nd=15)
                           2.8052381727677629	 % m=16 (nt=250, nd=15)
                           3.0839052775073315	 % m=20 (nt=250, nd=15)
                           3.4813169388381611	 % m=25 (nt=250, nd=15)
                           3.9380449652004392	 % m=30 (nt=250, nd=15)
                           4.5720229525229321	 % m=36 (nt=250, nd=15)
                           5.3097337895026042	 % m=42 (nt=250, nd=15)
                           6.3175099090303544	 % m=49 (nt=250, nd=15)
                           7.5040543573793697	 % m=56 (nt=250, nd=15)
                           9.1095320471991865]'; % m=64 (nt=250, nd=15)                    
                end             
                M=[2 4 6 9 12 16 20 25 30 36 42 49 56 64];
            case 'bernoulli' % Polinomio completo               
                % Resultados distintos para diferentes valores de nt
                nt=250;
                if nt==200
                    % Theta values of relative forward errors for cos or cosh (nt=200, nd=15)
                    theta=[0.0002271984505600	 % m= 2 (nt=200, nd=15)
                           0.0065633004324600	 % m= 4 (nt=200, nd=15)
                           0.0381353500337701	 % m= 6 (nt=200, nd=15)
                           0.1148773663474500	 % m= 9 (nt=200, nd=15)
                           0.4353426712417598	 % m=12 (nt=200, nd=15)
                           0.9520896293768095	 % m=16 (nt=200, nd=15)
                           1.5057811841086004	 % m=20 (nt=200, nd=15)
                           1.6432017599233411	 % m=25 (nt=200, nd=15)
                           1.7789273927034390	 % m=30 (nt=200, nd=15)
                           1.9228943496198816	 % m=36 (nt=200, nd=15)
                           2.0771839156390701	 % m=42 (nt=200, nd=15)                       
                           2.2421030188466822	 % m=49 (nt=200, nd=15)
                           2.4786858203387809	 % m=56 (nt=200, nd=15)
                           2.7344679640317202]'; % m=64 (nt=200, nd=15)
                elseif nt==250
                    % Theta values of relative forward errors for cos (nt=250, nd=15)                   
                    theta=[0.0002271984505600	 % m= 2 (nt=250, nd=15)
                           0.0065633004324600	 % m= 4 (nt=250, nd=15)
                           0.0381353500337701	 % m= 6 (nt=250, nd=15)
                           0.1148773663474500	 % m= 9 (nt=250, nd=15)
                           0.4353426712417598	 % m=12 (nt=250, nd=15)
                           0.9520896293768095	 % m=16 (nt=250, nd=15)
                           1.5057748070147008	 % m=20 (nt=250, nd=15)
                           1.6284719983562810	 % m=25 (nt=250, nd=15)
                           1.7387121860252199	 % m=30 (nt=250, nd=15)
                           1.8550422761995708	 % m=36 (nt=250, nd=15)
                           1.9791478475304800	 % m=42 (nt=250, nd=15)
                           2.1112279132154308	 % m=49 (nt=250, nd=15)
                           2.2997942557382922	 % m=56 (nt=250, nd=15)
                           2.5026354728187323]'; % m=64 (nt=250, nd=15)                   
                end
                M=[2 4 6 9 12 16 20 25 30 36 42 49 56 64]; 
            otherwise
                error('Método no válido para calcular el coseno');         
        end
    otherwise
         error('Tipo de error no contemplado');
end
end        


