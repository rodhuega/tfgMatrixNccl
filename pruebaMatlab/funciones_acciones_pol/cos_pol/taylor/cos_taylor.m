function E = cos_taylor(A,m)
% E = cos_taylor(A,m)
% Calcula cos(A) mediante la aproximación de Taylor.
%
% Datos de entrada:
% - A: Matriz.
% - m: Orden de la aproximación, de modo que el grado del polinomio a 
%      emplear será 2*m.
%
% Datos de salida:
% - E: Matriz resultado del coseno de A.

% POR COMPLETAR
E=0;
e=0;
for k=0:2:2*m
    % Los términos con A elevado a un valor impar valen 0, con lo cual
    % no los tenemos en cuenta
    E=E+(-1)^e*A^k/factorial(k);
    e=e+1;
end

end

