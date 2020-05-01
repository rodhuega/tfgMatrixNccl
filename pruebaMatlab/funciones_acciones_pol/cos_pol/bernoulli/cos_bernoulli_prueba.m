function fA=cos_bernoulli_prueba(A,m,s,formulacion,plataforma)
% La formulación podrá valer 'pares' o 'pares_impares'
% La plataforma podrá valer 'sinGPUs' o 'conGPUs'
q=ceil(sqrt(m));
pA{1}=A;
for i=2:q
    pA{i}=pA{i-1}*A;
end
pA=escalado_exp(plataforma,pA,s);
p = coefs_cos_bernoulli(m,formulacion);
[fA,nProd] = polyvalm_paterson_stockmeyer(plataforma,p,pA);
[fA,nProd]=escalado_regresivo_cos(plataforma,fA,s);

