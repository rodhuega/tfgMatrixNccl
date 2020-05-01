function set_formulacion_cos_bernoulli(formulacion)
% set_formulacion_cos_bernoulli(formulacion)
%
% Establece el valor de la variable global que determina la formulaci�n
% te�rica a emplear en el c�lculo del coseno mediante el m�todo de 
% Bernoulli. Podr� valer:
%   - 'terminos_pares_polinomio_completo': emplea la formulaci�n en la que 
%     s�lo se trabaja con t�rminos pares (aunque tambi�n pueden aparecer 
%     valores diferentes de 0 para las posiciones impares) y el polinomio 
%     se eval�a con todos sus t�rminos.
%   - 'terminos_pares_impares_polinomio_completo': emplea la formulaci�n en
%     la que se trabaja con t�rminos pares e impares y el polinomio se 
%     eval�a con todos sus t�rminos.
%   - 'terminos_pares_polinomio_solo_pares': emplea la formulaci�n en la 
%     que s�lo se trabaja con t�rminos pares (se entiende que los valores 
%     de las posiciones impares son muy peque�os y se desprecian, asumiendo
%     por tanto que valdr�n 0) y el polinomio se eval�a s�lo en los
%     t�rminos pares.


global formulacion_cos_bernoulli
formulacion_cos_bernoulli=formulacion;



        

