function [E,nProd] = polyvalm_exp_taylor_sastre(plataforma,p,pA)
% [E,nProd] = polyvalm_exp_taylor_sastre(plataforma,p,pA)
%
% Evaluación del polinomio matricial, mediante el método de Sastre, que
% proporciona la exponencial de una matriz A mediante Taylor.
%
% Datos de entrada:
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - p:          Vector (de m+1 elementos) con los coeficientes de la 
%               aproximación ordenados de mayor a menor grado.
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i} 
%               contiene A^i, para i=1,2,3,...,q.
%
% Datos de salida:
% - E:          Valor del polinomio p evaluado en A. Por compatibilidad, 
%               en el caso de la evaluación en la GPU se devolverá el 
%               vector vacío, ya que no se proporciona.
% - nProd:      Número de productos matriciales realizados por la función.

switch plataforma
    case 'sinGPUs'
        [E,nProd] = polyvalm_exp_taylor_sastre_sinGPUs(p,pA);
    case 'conGPUs'
        E=[];
        nProd = polyvalm_exp_taylor_sastre_conGPUs(p);
    otherwise
        error('Plataforma destino incorrecta');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [E,nProd] = polyvalm_exp_taylor_sastre_sinGPUs(p,pA)
% [E,nProd] = polyvalm_exp_taylor_sastre_sinGPUs(p,pA)
%
% Evaluación del polinomio matricial, mediante el método de Sastre, que
% proporciona la exponencial de una matriz A mediante Taylor, sin emplear
% GPUs.
%
% Datos de entrada:
% - p:     Vector (de m+1 elementos) con los coeficientes de la 
%          aproximación ordenados de mayor a menor grado.
% - pA:    Vector de celdas con las potencias de A, de modo que pA{i} 
%          contiene A^i, para i=1,2,3,...,q.
%
% Datos de salida:
% - E:     Valor del polinomio p evaluado en A. 
% - nProd: Número de productos matriciales realizados por la función.

n = size(pA{1});
m=length(p)-1; % Orden de la aproximación a usar
q=length(pA);  % Máxima potencia usada (A^q)

switch m
    case 1 %m=1 q=1
        E=pA{1}+eye(n);
        nProd = 0;
    case 2 %m=2 q=2
        E=pA{2}/2+pA{1}+eye(n);
        nProd = 0;
    case 3 %m=3 q=3 %Nilpotent matrices
        E=pA{3}/6+pA{2}/2+pA{1}+eye(n);
        nProd = 0;        
    case 4 %m=4 
        if q==2
            E=((pA{2}/4+pA{1})/3+eye(n))*pA{2}/2+pA{1}+eye(n);
            nProd=1;
        else %q=4 Nilpotent matrices
            E=((pA{4}/4+pA{3})/3+pA{2})/2+pA{1}+eye(n);
            nProd=0;
        end
    case 8 %m=8 q=2
        y0s=pA{2}*(p(1)*pA{2}+p(2)*pA{1});
        E=(y0s+p(3)*pA{2}+p(4)*pA{1})*(y0s+p(5)*pA{2})+p(6)*y0s+pA{2}/2+pA{1}+eye(n);
        nProd=2;
    case 15 %m=15 q=2
        y0s=pA{2}*(p(1)*pA{2}+p(2)*pA{1});
        y1s=(y0s+p(3)*pA{2}+p(4)*pA{1})*(y0s+p(5)*pA{2})+p(6)*y0s+p(7)*pA{2};
        E=(y1s+p(8)*pA{2}+p(9)*pA{1})*(y1s+p(10)*y0s+p(11)*pA{1})+p(12)*y1s+p(13)*y0s+p(14)*pA{2}+pA{1}+eye(n);
        nProd=3;
    case 21 %m=21 q=3
        y0s=pA{3}*(p(1)*pA{3}+p(2)*pA{2}+p(3)*pA{1});
        y1s=(y0s+p(4)*pA{3}+p(5)*pA{2}+p(6)*pA{1})*(y0s+p(7)*pA{3}+p(8)*pA{2})+p(9)*y0s+p(10)*pA{3}+p(11)*pA{2};
        E=(y1s+p(12)*pA{3}+p(13)*pA{2}+p(14)*pA{1})*(y1s+p(15)*y0s+p(16)*pA{1})+p(17)*y1s+p(18)*y0s+p(19)*pA{3}+p(20)*pA{2}+pA{1}+eye(n);
        nProd=3;
    case 24 %m=24 q=4
        y0s=pA{4}*(p(1)*pA{4}+p(2)*pA{3}+p(3)*pA{2}+p(4)*pA{1});
        y1s=(y0s+p(5)*pA{4}+p(6)*pA{3}+p(7)*pA{2}+p(8)*pA{1})*(y0s+p(9)*pA{4}+p(10)*pA{3}+p(11)*pA{2})+p(12)*y0s+p(13)*pA{4}+p(14)*pA{3}+p(15)*pA{2}+p(16)*pA{1};
        E=y1s*(y0s+p(17)*pA{4}+p(18)*pA{3}+p(19)*pA{2}+p(20)*pA{1})+p(21)*pA{4}+p(22)*pA{3}+p(23)*pA{2}+pA{1}+eye(n);
        nProd=3;
    case 30 %m=30 q=5
        y0s=pA{5}*(p(1)*pA{5}+p(2)*pA{4}+p(3)*pA{3}+p(4)*pA{2}+p(5)*pA{1});
        y1s=(y0s+p(6)*pA{5}+p(7)*pA{4}+p(8)*pA{3}+p(9)*pA{2}+p(10)*pA{1})*(y0s+p(11)*pA{5}+p(12)*pA{4}+p(13)*pA{3}+p(14)*pA{2})+p(15)*y0s+p(16)*pA{5}+p(17)*pA{4}+p(18)*pA{3}+p(19)*pA{2}+p(20)*pA{1};
        E=y1s*(y0s+p(21)*pA{5}+p(22)*pA{4}+p(23)*pA{3}+p(24)*pA{2}+p(25)*pA{1})+p(26)*pA{5}+p(27)*pA{4}+p(28)*pA{3}+p(29)*pA{2}+pA{1}+eye(n);
        nProd=3;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nProd = polyvalm_exp_taylor_sastre_conGPUs(p)
% nProd = polyvalm_exp_taylor_sastre_conGPUs(p)
%
% Evaluación del polinomio matricial, mediante el método de Sastre, que
% proporciona la exponencial de una matriz A mediante Taylor, mediante
% GPUs.
%
% - p:     Vector (de m+1 elementos) con los coeficientes de la 
%          aproximación ordenados de mayor a menor grado.
%
% Datos de salida:
% - nProd: Número de productos matriciales realizados.

nProd=call_gpu('evaluate',p);
end