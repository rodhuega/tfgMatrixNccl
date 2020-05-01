function set_formulacion_cos_bernoulli(formulacion)
% set_formulacion_cos_bernoulli(formulacion)
%
% Establece el valor de la variable global que determina la formulación
% teórica a emplear en el cálculo del coseno mediante el método de 
% Bernoulli. Podrá valer:
%   - 'terminos_pares_polinomio_completo': emplea la formulación en la que 
%     sólo se trabaja con términos pares (aunque también pueden aparecer 
%     valores diferentes de 0 para las posiciones impares) y el polinomio 
%     se evalúa con todos sus términos.
%   - 'terminos_pares_impares_polinomio_completo': emplea la formulación en
%     la que se trabaja con términos pares e impares y el polinomio se 
%     evalúa con todos sus términos.
%   - 'terminos_pares_polinomio_solo_pares': emplea la formulación en la 
%     que sólo se trabaja con términos pares (se entiende que los valores 
%     de las posiciones impares son muy pequeños y se desprecian, asumiendo
%     por tanto que valdrán 0) y el polinomio se evalúa sólo en los
%     términos pares.


global formulacion_cos_bernoulli
formulacion_cos_bernoulli=formulacion;



        

