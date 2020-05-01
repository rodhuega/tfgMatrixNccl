function [B,s,er]=pade_exp_vpa(A,nd)
digits(nd);
n=size(A,1);
er=0;

pA{1}=vpa(A);%A
if isinf(norm(pA{1}))
    er=1;
	B=[];
	return
end
pA{2}=A*A;%A^2
if isinf(norm(pA{2}))
    er=1;
	B=[];
	return
end
pA{3}=pA{2}*pA{2};%A^4
if isinf(norm(pA{3}))
    er=1;
	B=[];
	return
end
pA{4}=pA{2}*pA{3};%A^6
if isinf(norm(pA{4}))
    er=1;
	B=[];
	return
end

fin=0;
B=exp_pade(pA,n);
s=0;
while fin==0
    B0=B;
    s=s+1;
    pB{1}=pA{1}/2^s;
    pB{2}=pA{2}/2^(2*s); 
    pB{3}=pA{3}/2^(4*s);
    pB{4}=pA{4}/2^(6*s);
    B=exp_pade(pB,n);
    for k=1:s
        B=B*B;
    end
    er=norm(B-B0)/norm(B);
    %fprintf('s=%d: er=%g\n',s,er);
    if er<sym(eps/2)
        fin=1;
    end
end
end

function F = exp_pade(pA,n)
c = [64764752532480000, 32382376266240000, 7771770303897600, ...
        1187353796428800,  129060195264000,   10559470521600, ...
        670442572800,      33522128640,       1323241920,...
        40840800,          960960,            16380,  182,  1];

U = pA{1}*(pA{4}*(c(14)*pA{4} + c(12)*pA{3} + c(10)*pA{2}) ...
    + c(8)*pA{4} + c(6)*pA{3} + c(4)*pA{2} + c(2)*eye(n) );
V = pA{4}*(c(13)*pA{4} + c(11)*pA{3} + c(9)*pA{2}) ...
	+ c(7)*pA{4} + c(5)*pA{3} + c(3)*pA{2} + c(1)*eye(n);
F = (-U+V)\(U+V);
end
    