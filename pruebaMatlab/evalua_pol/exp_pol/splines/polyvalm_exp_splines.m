function [E,nProd] = polyvalm_exp_splines(plataforma,p,pA)
% [E,nProd] = polyvalm_exp_splines(p,pA)
%
% Evalúa el polinomio E=p(1)*I + p(2)*A + p(3)*A^2 + ...+ p(m+1)*A^m en A
% de forma eficiente por splines y por Paterson-Stockmeyer.
% La función sólo es válida para m= 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 
% 42, ..., es decir m=k*k o m=k*(k+1), k=1, 2, 3,...
%
% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - p:          Vector (de m+1 elementos) con los coeficientes del  
%               polinomio de menor a mayor grado.
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i} 
%               contiene A^i, para i=1,2,3,...,q. El valor de q se  
%               corresponde con q=ceil(sqrt(m)).
%
% Datos de salida:
% - E:          Valor del polinomio p evaluado en A. Por compatibilidad, 
%               en el caso de la evaluación en la GPU se devolverá el 
%               vector vacío, ya que no se proporciona.
% - nProd:      Número de productos matriciales realizados.


switch plataforma
    case 'sinGPUs'
        [E,nProd] = polyvalm_exp_splines_sinGPUs(p,pA);
    case 'conGPUs'
        E=[];
        nProd = polyvalm_exp_splines_conGPUs(p);
    otherwise
        error('Plataforma destino incorrecta');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [E,nProd] = polyvalm_exp_splines_sinGPUs(p,pA)
% [E,nProd] = polyvalm_exp_splines_sinGPUs(p,pA)
%
% Evalúa el polinomio E=p(1)*I + p(2)*A + p(3)*A^2 + ...+ p(m+1)*A^m en A
% de forma eficiente por splines y por Paterson-Stockmeyer sin emplear 
% GPUs.
% La función sólo es válida para m= 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 
% 42, ..., es decir m=k*k o m=k*(k+1), k=1, 2, 3,...
%
% Datos de entrada:
% - p:     Vector (de m+1 elementos) con los coeficientes del polinomio 
%          de menor a mayor grado.
% - pA:    Vector de celdas con las potencias de A, de modo que pA{i} 
%          contiene A^i, para i=1,2,3,...,q. El valor de q se corresponde 
%          con q=ceil(sqrt(m)).
%
% Datos de salida:
% - E:     Valor del polinomio p evaluado en A.
% - nProd: Número de productos matriciales realizados.

n=size(pA{1},1);
I=eye(n);
m=length(p)-1;
q=length(pA);

max_cond=1e1;
E=m*pA{1}-I;
nc=condest(E);
if nc<max_cond
    E=E\pA{q}*p(m);
    nProd=4/3;
else
    E=pA{q}*p(m+1);
    nProd=0;
end

% Implementación ligeramente modificada de Paterson-Stockmeyer
c=m;
k=m/q;
inic=q-1;
for j=k:-1:1
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nProd = polyvalm_exp_splines_conGPUs(p)
% nProd = polyvalm_exp_splines_conGPUs(p)
%
% Evalúa el polinomio E=p(1)*I + p(2)*A + p(3)*A^2 + ...+ p(m+1)*A^m en A
% de forma eficiente por splines y por Paterson-Stockmeyer mediante GPUs.
% La función sólo es válida para m= 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 
% 42, ..., es decir m=k*k o m=k*(k+1), k=1, 2, 3,...
%
% Datos de entrada:
% - p:     Vector (de m+1 elementos) con los coeficientes del polinomio 
%          de menor a mayor grado.
% - pA:    Vector de celdas con las potencias de A, de modo que pA{i} 
%          contiene A^i, para i=1,2,3,...,q. El valor de q se corresponde 
%          con q=ceil(sqrt(m)).
%
% Datos de salida:
% - nProd: Número de productos matriciales realizados.

nProd=call_gpu('evaluate',p);
end