function E = cosh_bernoulli(A,m,formulacion)
% E = cosh_bernoulli(A,m,formulcion)
% Calcula cosh(A) mediante la aproximación de Bernoulli empleando la serie
% en la cual aparecen sólo los términos pares o los términos pares e
% impares.
%
% Datos de entrada:
% - A:           Matriz.
% - m:           Orden de la aproximación, de modo que el grado del 
%                polinomio a emplear será 2*m.
% - formulación: Formulación teórica para trabajar sólo con los términos de 
%                posiciones pares ('pares') o con todos los términos
%                ('pares_impares').
%
% Datos de salida:
% - E:           Matriz resultado del coseno de A.

% PENDIENTE DE IMPLEMENTAR

switch formulacion
    case 'pares'
        E=0;
        n=size(A,1);
        A=(A+eye(n))/2;
        e=0;
        for k=0:2:2*m
            % Los términos con A elevado a un valor impar valen 0, con lo
            % cual no los tenemos en cuenta
            E=E+(pol_bernouilli(A,k)*2^k)/factorial(k);
            e=e+1;
        end
        E=sinh(1)*E;
    case 'pares_impares'
        E=0;
        % Trabajamos con los términos pares
        for k=0:m
            E=E+pol_bernouilli(A,2*k)/factorial(2*k);
        end
        E1=sinh(1)*E;
        E=0;
        % Trabajamos con los términos impares
        for k=0:m-1
            E=E+pol_bernouilli(A,2*k+1)/factorial(2*k+1);
        end
        E2=(cosh(1)-1)*E;
        E=E1+E2;        
end
end

