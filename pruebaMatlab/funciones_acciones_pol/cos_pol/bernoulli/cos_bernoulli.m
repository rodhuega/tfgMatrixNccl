function E = cos_bernoulli(A,m)
% E = cos_bernoulli(A,m)
% Calcula cos(A) mediante la aproximaci�n de Bernoulli empleando la serie
% en la cual aparecen s�lo los t�rminos pares o los t�rminos pares e
% impares.
%
% Datos de entrada:
% - A: Matriz.
% - m: Orden de la aproximaci�n, de modo que el grado del polinomio a 
%      emplear ser� 2*m.
%
% Datos de salida:
% - E: Matriz resultado del coseno de A.

% Obtenemos la formulaci�n te�rica a emplear
formulacion=get_formulacion_cos_bernoulli;

switch formulacion
    case 'terminos_pares_polinomio_completo'
        E=0;
        n=size(A,1);
        A=(A+eye(n))/2;
        e=0;
        for k=0:2:2*m
            % Los t�rminos con A elevado a un valor impar valen 0, con lo
            % cual no los tenemos en cuenta
            E=E+(-1)^e*(pol_bernouilli(A,k)*2^k)/factorial(k);
            e=e+1;
        end
        E=sin(1)*E;
    case 'terminos_pares_impares_polinomio_completo'
        E=0;
        % Trabajamos con los t�rminos pares
        for k=0:m
            E=E+(-1)^k*pol_bernouilli(A,2*k)/factorial(2*k);
        end
        E1=sin(1)*E;
        E=0;
        % Trabajamos con los t�rminos impares
        for k=0:m-1
            E=E+(-1)^k*pol_bernouilli(A,2*k+1)/factorial(2*k+1);
        end
        E2=(cos(1)-1)*E;
        E=E1+E2;
    otherwise
        error('Formulaci�n te�rica no contemplada');
end
end

