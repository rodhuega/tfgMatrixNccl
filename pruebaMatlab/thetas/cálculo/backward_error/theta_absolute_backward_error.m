function [theta,M]=theta_absolute_backward_error(f,nt,nd)
% [theta,M]=theta_absolute_backward_error(f,nt,nd)
%
% Cálcula los valores theta de la función f mediante series de Taylor y
% errores absolutos de tipo backward.
% 
% Datos de entrada:
% - f:     función (@exp, @cos, @cosh, @funcion_usuario). En el caso del
%          coseno con Taylor, donde sólo se trabaja con los términos pares,
%          definimos f=@(x)cos(sqrt(x)).
% - nt:    número de términos de la serie de Taylor (200, por ejemplo).
% - nd:    número de dígitos calculados para los valores de theta (15, por
%          ejemplo).
%
% Datos de salida:
% - theta: vector con los valores theta de la función f para los diferentes
%          grados que se especifican en el vector M.
% - M:     Vector con los grados del polinomio de aproximación para los
%          cuales calculamos el valor de theta.

% Grados del polinomio
M=[2 4 6 9 12 16 18 20 25 30 36 42 49 56 64];
theta=zeros(length(M),1);
fprintf('Theta values of absolute backward errors for %s (nt=%d, nd=%d)\n',func2str(f),nt,nd);
%for k=1:length(M)
for k=7:length(M)
    % En caso de valor impar, falla. Para evitarlo, sumamos uno más
    m=M(k);
    if rem(m,2)==1
        m=m+1;
    end
    eb = series_absolute_back_taylor(f,m,nt);
    x=0;inc=1;
    for i=1:nd
        while subs(eb,x)<eps/2
            x=x+inc;
        end
        x=x-inc;
        inc=inc/10;
    end
    theta(k)=x;
    fprintf('%.16f\t%% m=%2d (nt=%d, nd=%d)\n',x,M(k),nt,nd);
end
