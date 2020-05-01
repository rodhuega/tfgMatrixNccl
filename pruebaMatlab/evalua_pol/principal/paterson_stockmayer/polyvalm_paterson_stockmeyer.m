function [E,nProd] = polyvalm_paterson_stockmeyer(plataforma,p,pA)
% [E,nProd] = polyvalm_paterson_stockmeyer(plataforma,p,pA)
%
% Evalúa el polinomio E = p(1)*I + p(2)*A + p(3)*A^2 + ...+ p(m+1)*A^m en A
% de forma eficiente por Paterson-Stockmeyer. La función es válida para
% cualquier valor de m. El coste de la evaluación de un polinomio de grado
% m es equivalente al de un polinomio de grado mIdeal, siendo mIdeal un
% valor entero obtenido como el producto de k*k o de k*(k+1), con k=1, 2, 
% 3, ... Eso supone que mIdeal será el valor más cercano mayor o igual a m 
% del siguiente listado: 1, 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 49, 56,
% 64, etc. Ejemplo: si m=10, entonces mIdeal=12.

% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - p:          Vector (de m+1 elementos) con los coeficientes del 
%               polinomio de menor a mayor grado.
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i} 
%               contiene A^i, para i=1,2,3,...,q. El valor de q puede ser 
%               igual a k o a k+1, de modo que mIdeal=k*k o mIdeal=k*(k+1). 
%               En realidad, se corresponde con q=floor(sqrt(mIdeal)) o con 
%               q=ceil(sqrt(mIdeal)).
%               Si mIdeal=k*k, entonces q=k. Si mIdeal=k*(k+1), q=k o k+1.
%               Ejemplo: si mIdeal=25, entonces q=5. Si m=30, entonces q=5 
%               ó 6.
%
% Datos de salida:
% - E:          Valor del polinomio p evaluado en A. Por compatibilidad, 
%               en el caso de la evaluación en la GPU se devolverá el 
%               vector vacío, ya que no se proporciona.
% - nProd:      Número de productos matriciales realizados.

switch plataforma
    case 'sinGPUs'
        [E,nProd] = polyvalm_paterson_stockmeyer_sinGPUs(p,pA);
    case 'conGPUs'
        E=[];
        nProd = polyvalm_paterson_stockmeyer_conGPUs(p);
    otherwise
        error('Plataforma destino incorrecta');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [E,nProd] = polyvalm_paterson_stockmeyer_sinGPUs(p,pA)
% [E,nProd] = polyvalm_paterson_stockmeyer_sinGPUs(p,pA)
%
% Evalúa el polinomio E = p(1)*I + p(2)*A + p(3)*A^2 + ...+ p(m+1)*A^m en A
% de forma eficiente por Paterson-Stockmeyer sin emplear GPUs. La función 
% es válida para cualquier valor de m. El coste de la evaluación de un 
% polinomio de grado m es equivalente al de un polinomio de grado mIdeal, 
% siendo mIdeal un valor entero obtenido como el producto de k*k o de 
% k*(k+1), con k=1, 2, 3, ... Eso supone que mIdeal será el valor más 
% cercano mayor o igual a m del siguiente listado: 1, 2, 4, 6, 9, 12, 16, 
% 20, 25, 30, 36, 42, 49, 56, 64, etc. Ejemplo: si m=10 entonces mIdeal=12.

% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - p:          Vector (de m+1 elementos) con los coeficientes del 
%               polinomio de menor a mayor grado.
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i} 
%               contiene A^i, para i=1,2,3,...,q. El valor de q puede ser 
%               igual a k o a k+1, de modo que mIdeal=k*k o mIdeal=k*(k+1). 
%               En realidad, se corresponde con q=floor(sqrt(mIdeal)) o con 
%               q=ceil(sqrt(mIdeal)).
%               Si mIdeal=k*k, entonces q=k. Si mIdeal=k*(k+1), q=k o k+1.
%               Ejemplo: si mIdeal=25, entonces q=5. Si m=30, entonces q=5 
%               ó 6.
%
% Datos de salida:
% - E:          Valor del polinomio p evaluado en A. Por compatibilidad, 
%               en el caso de la evaluación en la GPU se devolverá el 
%               vector vacío, ya que no se proporciona.
% - nProd:      Número de productos matriciales realizados.

n=size(pA{1},1);
I=eye(n);
m=length(p)-1;
c=m+1;
q=length(pA);
%k=m/q;
k=ceil(m/q);
mIdeal=q*k;

E=zeros(n);
nProd=0;
for j=k:-1:1
    if j==k
        %inic=q;
        inic=q-mIdeal+m;
    else
        inic=q-1;
    end
    for i=inic:-1:1
        E=E+p(c)*pA{i};
        c=c-1;
    end
    E=E+p(c)*I;
    c=c-1;
    if j~=1
        E=E*pA{q}; 
        nProd=nProd+1;
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nProd = polyvalm_paterson_stockmeyer_conGPUs(p)
% nProd = polyvalm_paterson_stockmeyer_conGPUs(p)
%
% Evalúa el polinomio E = p(1)*I + p(2)*A + p(3)*A^2 + ...+ p(m+1)*A^m en A
% de forma eficiente por Paterson-Stockmeyer mediante GPUs.
% La función sólo es válida para m=1, 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 
% 42, ..., es decir m=k*k o m=k*(k+1), k=1, 2, 3,...
% En el caso de órdenes m=k*k, el valor de q sería igual a k.
% En el caso de órdenes m=k*(k+1), el valor de q puede ser k o k+1.
% El resultado de la evaluación no se proporciona.
% Datos de entrada:
% - p:     Vector (de m+1 elementos) con los coeficientes del polinomio 
%          de menor a mayor grado.
%
% Datos de salida:
% - nProd: Número de productos matriciales realizados.

nProd=call_gpu('evaluate',p);
end
