function [E,nProd] = polyvalm_cos_taylor_sastre(plataforma,p,pB)
% [E,nProd] = polyvalm_cos_taylor_sastre(plataforma,p,pB)
%
% Evaluación del polinomio matricial, mediante el método de Sastre, que
% proporciona el coseno de una matriz A mediante Taylor.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - p:          Vector (de m+1 elementos) con los coeficientes de la 
%               aproximación ordenados de mayor a menor grado.
% - pB:         Vector de celdas con las potencias de B=A^2, de modo que 
%               pB{i} contiene B^i, para i=1,2,3,...,q, siendo q<=4.
%
% Datos de salida:
% - E:          Valor del polinomio p evaluado en A. Por compatibilidad, 
%               en el caso de la evaluación en la GPU se devolverá el 
%               vector vacío, ya que no se proporciona.
% - nProd:      Número de productos matriciales realizados por la función.

switch plataforma
    case 'sinGPUs'
        [E,nProd] = polyvalm_cos_taylor_sastre_sinGPUs(p,pB);
    case 'conGPUs'
        E=[];
        nProd = polyvalm_cos_taylor_sastre_conGPUs(p);
    otherwise
        error('Plataforma destino incorrecta');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [E,nProd] = polyvalm_cos_taylor_sastre_sinGPUs(p,pB)
% [E,nProd] = polyvalm_cos_taylor_sastre_sinGPUs(p,pB)
%
% Evaluación del polinomio matricial, mediante el método de Sastre, que
% proporciona el coseno de una matriz A mediante Taylor, sin emplear GPUs.
%
% Datos de entrada:
% - p:     Vector (de m+1 elementos) con los coeficientes de la 
%          aproximación ordenados de mayor a menor grado.
% - pB:    Vector de celdas con las potencias de B=A^2, de modo que pB{i} 
%          contiene B^i, para i=1,2,3,...,q, siendo q<=4.
%
% Datos de salida:
% - E:     Valor del polinomio p evaluado en A.
% - nProd: Número de productos matriciales realizados por la función.

n=size(pB{1},1);
m=length(p)-1; % Orden de la aproximación a usar

switch m
    case 0
        E = eye(n);
        nProd = 0;
    case 1
        E = -pB{1}/2 + eye(n);
        nProd = 0;
    case 2
        E = pB{2}/24 - pB{1}/2 + eye(n);
        nProd = 0;        
    case 4
        E = pB{2}*(pB{2}/40320 - pB{1}/720) + pB{2}/24 - pB{1}/2 + eye(n);
        nProd = 1;                
    case 8
        y0=pB{2} * (p(1)*pB{2}+p(2)*pB{1});
        E=(y0+p(3)*pB{2}+p(4)*pB{1}) * (y0+p(5)*pB{2})+p(6)*y0+pB{2}/24-pB{1}/2+eye(n);
        nProd = 2;
    case 12
        y0=pB{3} * (p(1)*pB{3}+p(2)*pB{2}+p(3)*pB{1});
        E=(y0+p(4)*pB{3}+p(5)*pB{2}+p(6)*pB{1}) * (y0+p(7)*pB{3}+p(8)*pB{2})+p(9)*y0+p(10)*pB{3}+pB{2}/24-pB{1}/2+eye(n);
        nProd = 2;        
    case 15
        y0=pB{3} * (p(1)*pB{3}+p(2)*pB{2}+p(3)*pB{1});
        E=-((y0+p(4)*pB{3}+p(5)*pB{2}+p(6)*pB{1}) * (y0+p(7)*pB{3}+p(8)*pB{2})+p(9)*y0+p(10)*pB{3}+pB{2}/3628800-pB{1}/40320+eye(n)/720)*pB{3}+pB{2}/24-pB{1}/2+eye(n);
        nProd = 3;        
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nProd = polyvalm_cos_taylor_sastre_conGPUs(p)
% nProd = polyvalm_cos_taylor_sastre_conGPUs(p)
%
% Evaluación del polinomio matricial, mediante el método de Sastre, que
% proporciona el coseno de una matriz A mediante Taylor, mediante GPUs.
%
% Datos de entrada:
% - p:     Vector (de m+1 elementos) con los coeficientes del polinomio
%          ordenados de mayor a menor grado.
%
% Datos de salida:
% - nProd: Número de productos matriciales realizados.

nProd=call_gpu('evaluate',p);
end