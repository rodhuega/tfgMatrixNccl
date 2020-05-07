 Theta=[
      2.580956802971767e-08  %m=2
      0.00033971688399769617 %m=4
      0.0090656564075951018  %m=6
      0.08957760203223343    %m=9
      0.29961589138115807    %m=12
      0.78028742566265741    %m=6
      1.4382525968043369     %m=20
      2.4285825244428265     %m=25
      3.5396663487436895     %m=30
      4.9729156261919814     %m=36
      6.4756827360799845     %m=42
      8.2848536298039175     %m=49
      10.133428317897478     %m=56
      12.277837026932959];   %m=64
M= [2 4 6 9 12 16 20 25 30 36 42 49 56 64];
q=[2 2 3 3  4  4  5  5  6  6  7  7  8  8];
pA{1}=A;
nofin=1;im=0;imax=14;
while nofin&&im<imax
    im=im+1;
    j=ceil(sqrt(M(im)));
    if sqrt(M(im))>floor(sqrt(M(im)))
        pA{j}=pA{j-1}*A;
    end
    alfa(im)=norm1pp(pA{q(im)},q(im),A)^(1/(M(im)+1));
    if alfa(im)<Theta(im)
        nofin=0;     
    end
end
if nofin==0
    sm=0;
else
    sm=ceil(max(0,log2(alfa(im)/Theta(im))));
end  
m=M(im);
s=sm;