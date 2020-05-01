function genera_matrices_diag_hadamard_complex(F,n,nmat,nd)
% genera_matrices_diag_hadamard_complex(F,n,nmat,nd)
%
% Genera matrices de test diagonalizables con transformaciones ortogonales
% a partir de matrices V de Hadamard, con valores propios reales y
% complejos de magnitud creciente. Por cada matriz se genera un fichero 
% .mat que almacena:
% - A:          La propia matriz, generada como V*D*V'. 
% - normA:      1-norma de la matriz A.
% - condV:      N�mero de condici�n de la matriz V (vectores propios).
% - condFA:     N�mero de condici�n de la funci�n matricial.
% - FA:         Resultado de la funci�n matricial aplicada sobre la matriz A.
% - flag_error: Indica si ha ocurrido alg�n error. Valdr� 0 (no hay error),
%               -2 (ha ocurrido un error en el c�lculo de la matriz A y 
%               A valdr� infinito o not a number), -3 (ha ocurrido un error 
%               en el c�lculo de la funci�n matricial y FA valdr� infinito 
%               o not a number). 
% Adicionalmente, se crea un fichero .m con el n�mero de matrices
% generadas.
%
% Datos de entrada:
% - F:    Funci�n a aplicar sobre la matriz (@exp, @cos, @cosh, ...).
% - n:    N�mero de filas y columnas de la matriz. Para ver los tama�os
%         posibles, consultar las restricciones de la generaci�n de las
%         matrices de Hadamard (n, n/12 or n/20 deben ser potencias de 2).
% - nmat: N�mero de matrices a generar.
% - nd:   N�mero de d�gitos vpa (32, 64, 128, 256).
%
% Ejemplo de invocaci�n:
% genera_matrices_diag_hadamard_complex(@exp,128,100,256)
%
% Referencia:
%   P�gina 98 de la tesis de Javier para calcular F([a b; c d])

t0=sprintf('%s_diag_hadamard_complex_n%d_nd%d',func2str(F),n,nd);
try
    warning off
    eval(sprintf('mkdir %s',t0));
catch
end
digits(nd);
rng('default');
P=vpa(hadamard(n)/sqrt(n));
for k=1:nmat    
	% Generamos una matriz Ad diagonal a bloques de tama�o 1x1 o 2x2.
	% Generamos tambi�n expAd=F(Ad), siendo F la funci�n exponencial
	Ad=zeros(n);
	FAd=zeros(n);
	tic;
	j=1;
	while j<=n
        if rand()<0.5 && j<n % Generamos un bloque 2x2 
            % Completamos un bloque de tama�o 2x2 en la diagonal
            % de la matriz Ad de la forma [a b; c a], siendo c=-b.
            a=vpa((rand()-0.5)*k/2); 
            b=vpa((rand()-0.5)*k/2); 
            c=-b;
            % Generaci�n de la matriz Ad
            Ad(j,j)=a; 
            Ad(j,j+1)=b;
            Ad(j+1,j)=c; 
            Ad(j+1,j+1)=a;                    
            % Generaci�n de la matriz FAd=f(Ad)=F([a b; c a])
            z=a+sqrt(b*c); % N�mero complejo z
            fz=F(z);       % N�mero complejo F(z)
            rfz=real(fz);  % Parte real de F(z)
            ifz=imag(fz);  % Parte imaginaria de F(z) 
            aux=sqrt(-b*c)*ifz;
            FAd(j,j)=rfz;
            FAd(j,j+1)=-aux/c;
            FAd(j+1,j)=-aux/b;
            FAd(j+1,j+1)=rfz;
            j=j+2;
        else % Generamos un bloque 1x1
            % a=vpa((rand()-0.5)*k/2); % Dividimos tambi�n k por 2
            a=vpa((rand()-0.5)*k); % No dividimos por 2 para compensar con el caso anterior
            fa=F(a); % fa=F(a)=F(a)
            % Generaci�n de la matriz Ad
            Ad(j,j)=a;
            % Generaci�n de la matriz FAd=F(Ad)=F(a)
            FAd(j,j)=fa;
            j=j+1;            
        end
    end
	% A partir de las matrices P, Ad y FAd, obtenemos A y FA
	A=P*Ad*P';
	FA=P*FAd*P'; 
	clear Ad FAd
	A=double(A);
	FA=double(FA);
	normA=norm(A,1);
	condFA=funm_condest1(A,F);
	condV=cond(double(P));
	if max(max((isnan(A)))) || max(max((isinf(A))))
        flag_error=-2;        
        fprintf('La matriz %d no se ha podido calcular (A=Inf o A=NaN)\n',k);
    elseif max(max((isnan(FA))))||max(max((isinf(FA))))
        flag_error=-3;
        fprintf('La funci�n %s no se ha podido aplicar sobre la matriz %d (FA=Inf o FA=NaN)\n',func2str(F),k);
    else
        flag_error=0;
    end
	% Nombre del fichero de salida
	t=sprintf('save %s/%s_%d.mat A FA normA condV condFA flag_error',t0,t0,k);
	eval(t);
    fprintf('Matriz %d (norm(A,1)=%f, cond(V)=%f): %f segundos\n',k,normA,condV,toc)
end
t=sprintf('%s/%s.m',t0,t0);
fileID = fopen(t,'w');
fprintf(fileID,'nmat=%d;',nmat);
fclose(fileID);
end