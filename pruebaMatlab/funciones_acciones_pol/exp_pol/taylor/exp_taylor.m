function E = exp_taylor(A,m)
% E = exp_taylor(A,m)
% Calcula e^A mediante la aproximación de Taylor.
%
% Datos de entrada:
% - A: Matriz.
% - m: Orden de la aproximación. Coincide con el grado del polinomio a 
%      emplear.
%
% Datos de salida:
% - E: Matriz resultado de la exponencial de A.

E=0;
for k=0:m
    E=E+A^k/factorial(k);
end

end

