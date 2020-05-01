function Bn = pol_bernouilli(A,n)
% Bn = pol_bernouilli(A,n)
% Proporciona el polinomio n-ésimo de Bernoulli evaluado sobre la matriz
% A, es decir, Bn(A).
% Datos de entrada:
% - A: Matriz.
% - n: Orden del polinomio.
% Datos de salida:
% - Bn: Matriz resultante.

% Obtenemos los n+1 números primeros de Bernoulli mediante la función de 
% Matlab
B=bernoulli(0:n);
% Implementación del sumatorio
Bn=0;
for k=0:n
    Bn=Bn+nchoosek(n,k)*B(k+1)*A^(n-k);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function B=num_bernoulli(k)
% b=num_bernoulli(k)
% Proporciona el número k de Bernoulli, como alternativa a la función 
% bernoulli de Matlab.
if k==0
    B=1;
elseif (rem(k,2)~=0 && k>1)
    B=0;
else
    B=0;
    for i=0:k-1
        B=B+nchoosek(k,i)*num_bernoulli(i)/(k+1-i);
    end
    B=-B;
end
end






