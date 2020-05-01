function [theta,M]=get_theta_cos_backward(metodo_f,tipo_error)
%[theta,M]=get_theta_cos_backward(metodo_f,tipo_error)
%
% Proporciona los valores theta de la función coseno o coseno hiperbólico
% ante errores absolutos o relativos de tipo backward.
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
                % Theta values of absolute backward errors for @(x)cos(sqrt(x)) (nt=100, nd=15)
                theta=[0.0000341903468800	% m= 2 (nt=100, nd=15)
                       0.0114992545416500	% m= 4 (nt=100, nd=15)
                       0.1733508225353601	% m= 6 (nt=100, nd=15)
                       2.6128136156806216	% m= 9 (nt=100, nd=15)
                       5.6861650846970262   % m=12 (nt=100, nd=15)
                      10.6451338629434851   % m=16 (nt=100, nd=15)
                      %11.4286532095767424	% m=18 (nt=100, nd=15)
                      12.5498932956960161]';% m=20 (nt=100, nd=15)                  
                %M=[2 4 6 9 12 16 18 20];% 25 30 36 42 49 56 64]; 
                M=[2 4 6 9 12 16 20];% 25 30 36 42 49 56 64]; 
            case 'bernoulli' % Polinomio completo
                % Theta values of absolute backward errors for cos (nt=150, nd=15)                
                %theta=[0.0000138635299100	 % m= 2 (nt=150, nd=15)
                %       0.0024018643198800	 % m= 4 (nt=150, nd=15)
                %       0.0239165687595000	 % m= 6 (nt=150, nd=15)
                %       0.2179723772395201	 % m= 9 (nt=150, nd=15)
                %       0.4105569044164998	 % m=12 (nt=150, nd=15)
                %       0.9709449019556995	 % m=16 (nt=150, nd=15)
                %       1.7043994615938596	 % m=20 (nt=150, nd=15)
                %       2.8941775730192405	 % m=25 (nt=150, nd=15)
                %       3.2469886161188324	 % m=30 (nt=150, nd=15)
                %       3.4716897717712412	 % m=36 (nt=150, nd=15)
                %       3.7584546403369092	 % m=42 (nt=150, nd=15)
                %       4.2419415127133862	 % m=49 (nt=150, nd=15)
                %       4.4388733626596784	 % m=56 (nt=150, nd=15)
                %       4.6700938252567994]';% m=64 (nt=150, nd=15)
                % Theta values of absolute backward errors for cos (nt=200,nd=15)
                % theta=[0.0000138635299100	% m= 2 (nt=200, nd=15)
                %       0.0024018643198800	% m= 4 (nt=200, nd=15)
                %       0.0239165687595000	% m= 6 (nt=200, nd=15)
                %       0.2179723772395201	% m= 9 (nt=200, nd=15)
                %       0.4105569044164998	% m=12 (nt=200, nd=15)
                %       0.9709449019556995	% m=16 (nt=200, nd=15)
                %       1.7043994615938596	% m=20 (nt=200, nd=15)
                %       2.8941773491997105	% m=25 (nt=200, nd=15)
                %       3.2156493760415117	% m=30 (nt=200, nd=15)
                %       3.4202884888348524	% m=36 (nt=200, nd=15)
                %       3.6450792416680011	% m=42 (nt=200, nd=15)                       
                %       3.9909644233489114	% m=49 (nt=200, nd=15)
                %       4.1389122930049620	% m=56 (nt=200, nd=15)
                %       4.3109370027999301]';% m=64 (nt=200, nd=15)
                % Theta values of absolute backward errors for cos (nt=250, nd=15)
                theta=[0.0000138635299100	% m= 2 (nt=250, nd=15)
                       0.0024018643198800	% m= 4 (nt=250, nd=15)
                       0.0239165687595000	% m= 6 (nt=250, nd=15)
                       0.2179723772395201	% m= 9 (nt=250, nd=15)
                       0.4105569044164998	% m=12 (nt=250, nd=15)
                       0.9709449019556995	% m=16 (nt=250, nd=15)
                       1.7043994615938596	% m=20 (nt=250, nd=15)
                       2.8941773460562614	% m=25 (nt=250, nd=15)
                       3.1977957144598710	% m=30 (nt=250, nd=15)
                       3.3824930164630418	% m=36 (nt=250, nd=15)
                       3.5665559850223501	% m=42 (nt=250, nd=15)
                       3.8325140285675636	% m=49 (nt=250, nd=15)
                       3.9508114735029514	% m=56 (nt=250, nd=15)
                       4.3109370027999301]';% m=64 (nt=250, nd=15)                   
                M=[2 4 6 9 12 16 20 25 30 36 42 49 56 64];                
            otherwise
                error('Método no válido para calcular el coseno'); 
        end
    case 'relativo'
        switch metodo_f
            case {'taylor','hermite'}
                % Theta values of relative backward errors for @(x)cos(sqrt(x)) (nt=100, nd=15)
                theta=[0.0000001999200500	 % m= 2 (nt=100, nd=15)
                       0.0037667842359700    % m= 4 (nt=100, nd=15)
                       0.1295985844455301    % m= 6 (nt=100, nd=15)
                       2.8614895493851735    % m= 9 (nt=100, nd=15)
                       6.4367291259772790    % m=12 (nt=100, nd=15)
                      10.8854864204769992    % m=16 (nt=100, nd=15)
                      %11.6534165387044517    % m=18 (nt=100, nd=15)                  
                      12.8161690859999915    % m=20 (nt=100, nd=10) 
                      16.1786015120000286]'; % m=25 (nt=100, nd=10)                  
                %M=[2 4 6 9 12 16 18 20 25];% 30 36 42 49 56 64];                 
                M=[2 4 6 9 12 16 20 25];% 30 36 42 49 56 64];                 
            case 'bernoulli' % Polinomio completo
                % Theta values of relative backward errors for cos (nt=150, nd=15)
                %theta=[0.0000000516191300	 % m= 2 (nt=150, nd=15)
                %       0.0005317232823300	 % m= 4 (nt=150, nd=15)
                %       0.0128376816920900	 % m= 6 (nt=150, nd=15)
                %       0.1872100253916101	 % m= 9 (nt=150, nd=15)
                %       0.3813222151367199	 % m=12 (nt=150, nd=15)
                %       0.9691933776789498	 % m=16 (nt=150, nd=15)
                %       1.7476885455576903	 % m=20 (nt=150, nd=15)
                %       2.9703027073639401	 % m=25 (nt=150, nd=15)
                %       3.2721747075580612	 % m=30 (nt=150, nd=15)
                %       3.4928062704227711	 % m=36 (nt=150, nd=15)
                %       3.7829923618568415	 % m=42 (nt=150, nd=15)
                %       4.2733939312959892	 % m=49 (nt=150, nd=15)
                %       4.4690441124305602	 % m=56 (nt=150, nd=15)
                %       4.6987010860543572]';% m=64 (nt=150, nd=15)
                % Theta values of relative backward errors for cos (nt=200, nd=15)
                %theta=[0.0000000516191300	 % m= 2 (nt=200, nd=15)
                %       0.0005317232823300	 % m= 4 (nt=200, nd=15)
                %       0.0128376816920900	 % m= 6 (nt=200, nd=15)
                %       0.1872100253916101	 % m= 9 (nt=200, nd=15)
                %       0.3813222151367199	 % m=12 (nt=200, nd=15)
                %       0.9691933776789498	 % m=16 (nt=200, nd=15)
                %       1.7476885455576903	 % m=20 (nt=200, nd=15)
                %       2.9702917684546608	 % m=25 (nt=200, nd=15)
                %       3.2343336923329291	 % m=30 (nt=200, nd=15)
                %       3.4378918294259617	 % m=36 (nt=200, nd=15)
                %       3.6644251439136615	 % m=42 (nt=200, nd=15)
                %       4.0135931430694578	 % m=49 (nt=200, nd=15)
                %       4.1603214354764297	 % m=56 (nt=200, nd=15)
                %       4.3309319482197406]';% m=64 (nt=200, nd=15)
                % Theta values of relative backward errors for cos (nt=250, nd=15)
                theta=[0.0000000516191300	 % m= 2 (nt=250, nd=15)
                       0.0005317232823300	 % m= 4 (nt=250, nd=15)
                       0.0128376816920900	 % m= 6 (nt=250, nd=15)
                       0.1872100253916101	 % m= 9 (nt=250, nd=15)
                       0.3813222151367199	 % m=12 (nt=250, nd=15)
                       0.9691933776789498	 % m=16 (nt=250, nd=15)
                       1.7476885455576903	 % m=20 (nt=250, nd=15)
                       2.9702911762904001	 % m=25 (nt=250, nd=15)
                       3.2126861576052130	 % m=30 (nt=250, nd=15)
                       3.3975794188593009	 % m=36 (nt=250, nd=15)
                       3.5824928543070507	 % m=42 (nt=250, nd=15)
                       3.8500703385834818	 % m=49 (nt=250, nd=15)
                       3.9672646268255405	 % m=56 (nt=250, nd=15)
                       4.3309319482197406]'; % m=64 (nt=250, nd=15)                   
                M=[2 4 6 9 12 16 20 25 30 36 42 49 56 64]; 
            otherwise
                error('Método no válido para calcular el coseno');         
        end
    otherwise
         error('Tipo de error no contemplado');
end
end        


