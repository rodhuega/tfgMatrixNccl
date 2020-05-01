function max_min_normas(dir)
% max_min_normas(dir)
%
% Calcula y muestra por pantalla la mínima y la máxima 1-norma de las 
% matrices contenidas en la carpeta indicada por la variable dir.
%
% Datos de entrada:
% - dir: Cadena de caracteres con el nombre del directorio en el que se 
%        encuentran los ficheros con las matrices.
% 
% Ejemplo de invocación: max_min_normas('exp_diag_hadamard_real_n128_maxv5_nd256')

eval(dir);
max_norm=0;
min_norm=realmax;
for k=1:nmat
    tic
    t=sprintf('load %s_%d.mat A FA normA condFA condV',dir,k);
	eval(t);
    if normA>max_norm
        max_norm=normA;
    end
    if normA<min_norm
        min_norm=normA;
    end
end
fprintf('Mínima norma=%g.\tMáxima norma=%g\n',min_norm,max_norm)
end