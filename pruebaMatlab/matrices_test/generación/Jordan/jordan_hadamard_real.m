function [A,FA,condV]=jordan_hadamard_real(F,n,bound_vp,max_mult,nd)
% [A,FA,condV]=jordan_hadamard_real(F,n,bound_vp,max_mult,nd)
%
% Genera una matriz de Jordan a partir de matrices V de Hadamard con 
% valores propios reales comprendidos en un intervalo concreto y con una
% multiplicidad determinada. 
%
% Datos de entrada:
% - F:        Función a aplicar sobre la matriz (@exp, @cos, @cosh, ...).
% - n:        Número de filas y columnas de la matriz. Para ver los tamaños
%             posibles, consultar las restricciones de la generación de las
%             matrices de Hadamard (n, n/12 or n/20 deben ser potencias de 
%             2).
% - bound_vp: Valor que delimina el intervalo ]-bound_vp, bound_vp[ en el 
%             que se encuentran los valores propios de la matriz.
% - max_mult: Determina el intervalo [1, max_mult] en el que se encuentran
%             las multiplicidades de los valores propios.
% - nmat:     Número de matrices a generar.
% - nd:       Número de dígitos vpa (32, 64, 128, 256).
%
% Datos de salida:
% - A:        Matriz, generada como V*D*V'.
% - FA:       Resultado de la función matricial aplicada sobre la matriz A.
% - condV:    Número de condición de la anterior matriz V (vectores 
%             propios).

% Ejemplo de invocación:
% [A,FA,condV]=jordan_hadamard_real(@exp,128,10,5,256)

digits(nd);
k=0;
s=0;
nofin=1;
while nofin
    k=k+1;
    m(k)=floor(max_mult*rand)+1;
    v(k)=vpa(bound_vp*(2*rand-1));
    s=s+m(k);
    nofin=s<n;
end
m(k)=m(k)-(s-n);
[As,FAs]=blocks_Jordan_F(F,v,m,nd);
V=vpa(vpa(hadamard(n))/vpa(sqrt(n)));
A=vpa(vpa(V*As)*V');
FA=(vpa(V*FAs)*V');
A=double(A);
FA=double(FA);
condV=cond(double(V));


