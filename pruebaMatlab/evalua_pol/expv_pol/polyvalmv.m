function fAv = polyvalmv(plataforma,p,A,v,s)
% fAv = polyvalmv(plataforma,p,A,v,s)
% 
% Calcula el producto fAv=(p(1)*I + p(2)*A + p(3)*A^2 + ...+ p(m+1)*A^m)*v.
% de forma eficiente, realizando productos matriz por vector.
%
% Datos de entrada:
% - plataforma: Decide si calculamos el producto mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - p:          Vector (de m+1 elementos) con los coeficientes del 
%               polinomio de menor a mayor grado.
% - A:          Matriz a calcular su exponencial expresada en forma de un
%               array de celdas de una única componente.
% - v:          Vector columna, de dimensión igual a A, por el que
%               multiplicamos el polinomio.
% - s:          Valor del escalado aplicado sobre la matriz.
%
% Datos de salida:
% - fAv:        Resultado del producto del polinomio por v.

if s==0
    s=1;
end

m=length(p)-1;

for i = 1:s
   fAv = p(1)*v;
   for j = 1:m
       v = A{1}*v;
       fAv =  fAv + p(j+1)*v;
   end
   v = fAv;        
end

end

