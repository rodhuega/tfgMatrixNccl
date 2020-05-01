function E = exp_bernoulli(A,m,L)
% E = exp_bernoulli(A,m,L,s)
% Calcula e^A mediante la aproximación de Bernoulli, empleando la técnica 
% de scaling and squaring sobre t.
%
% Datos de entrada:
% - A: Matriz.
% - m: Orden de la aproximación.
% - L: Lambda o escalado de la variable t.
%
% Datos de salida:
% - E: Matriz resultado de la exponencial de A.

E=exp_bernoulli_L(A,m,L);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function E = exp_bernoulli_L(A,m,L)
% E = exp_bernoulli_L(A,m,L)
% Calcula e^A mediante los polinomios de Bernoulli empleando escalado 
% sobre la variable t=1/L, es decir:
% e^A=[e^(A/L)]^L
% Datos de entrada:
% - A: Matriz.
% - L: Lambda o escalado de la variable t.
% Datos de salida:
% - E: Matriz resultado de la exponencial de A.
E=0;
for n=0:m
    E=E+pol_bernouilli(A,n)/(factorial(n)*L^n);
end
E=L*(exp(1/L)-1)*E;
% Deshacemos el escalado
E=E^L;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

