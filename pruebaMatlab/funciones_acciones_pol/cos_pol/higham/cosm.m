function [C,m,s,nprods] = cosm(A, schur_fact)
% Modificado de COSM  de Higham con los mismos argumentos de salida que
% los códigos implementados
% This code computes the matrix cosine of a given matrix A using
% a rational approximant based upon the exponential.
%
% The second (optional) argument SCHUR_FACT can be one of the following:
% 0 - No Schur decomposition used (this is the default).
% 1 - Real Schur decomposition used where possible.
% 2 - Always use the complex Schur decomposition.
%
% Please see the corresponding paper for when use of the Schur
% decomposition is beneficial but essentially it is:
% - potentially more accurate when A is nonnormal.
% - faster when A is sufficiently large and nonnormal.
%
% This code is intended for double precision arithmetic.
%
% Reference: A. H. Al-Mohy, N. J. Higham, and Samuel D. Relton,
% New Algorithms for the Matrix Sine and Cosine Separately or
% Simultaneously. SIAM J. Sci. Comput., 37(1), A456-A487, 2015.
%
% Corresponding algorithm from the paper: Algorithms 4.1 & 4.2
%
% Author: N. J. Higham and Samuel D. Relton.
% Date: September 22, 2015.

%% Check input parameters.
if nargin < 1
    error('cosm:NoInput', 'Not enough input parameters to cosm.')
end
if nargin < 2
    schur_fact = 0; % Default no Schur decomposition
end

if nargin == 2 && ~(schur_fact == 0 || schur_fact == 1 || schur_fact == 2)
    error('cosm:InvalidSchur', 'SCHUR_FACT argument is invalid.')
end

% Check matrix A is valid.
if ~ismatrix(A) || any(any(~isfinite(A))) || any(any(isnan(A)))
    error('cosm:NotMatrix', 'Input A must be a finite matrix.')
end

[rows, n] = size(A);
if rows ~= n
    error('cosm:Rectangular', 'Input matrix A must be square.')
end

%% Perform initialization.
useschur = false; % Use initial Schur decomposition.
transposed = false; % Do we need to transpose the matrix?
triang = true; % Will we be working on a triangular matrix?

if istril(A)
    transposed = true;
    T = transpose(A);
elseif istriu(A)
    T = A;
elseif schur_fact == 1
    useschur = true;
    [Q, T] = schur(A);
elseif schur_fact == 2
    useschur = true;
    [Q, T] = schur(A, 'complex');
else
    T = A; % Operate on full matrix
    triang = false;
end

%% Main computation
[s, m, Tpowers] = parameter_selection(T); % Find optimal parameters

% Scaling down
for k = 2:2:length(Tpowers)
    Tpowers{k} = Tpowers{k}/2^(s*k);
end

[C,nprods]= rational_approx(Tpowers, m); % Perform rational approximation

if triang
    C = recompute_diag(C, T, s); % Recompute 2x2 blocks on diagonal
end

% Do 'squaring' phase.
I = eye(n);
for k = 1:s
    C = 2*C^2 - I;
    if triang
        C = recompute_diag(C, T, s-k); % Recompute 2x2 blocks on diagonal
    end
end
nprods=nprods+s;


%% Finalize
if transposed
    C = transpose(C);
end

if useschur
    C = Q*C*Q';
    nprods=nprods+14;
end

end % of cosm_new

%% Subfunctions
function [s, m, Tpowers] = parameter_selection(T)
% PARAMETER_SELECTION Chooses optimal parameters for the matrix cosine.
%
% This function selects the optimal parameters for:
% S - the number of scalings.
% M - the degree of rational approximant used.
%
% We also compute some values of T^k, stored in Tpowers,
% which are used later when forming the rational approximant.

   th = [3.650024139523051e-08,...
       5.317232856892575e-04,...
       1.495585217958291e-02,...
       8.536352760102744e-02,...
       2.539398330063230e-01,...
       5.414660951208968e-01,...
       9.504178996162932e-01,...
       1.473163964234804e+00,...
       2.097847961257068e+00,...
       2.811644121620263e+00,...
       3.602330066265032e+00,...
       4.458935413036850e+00,...
       5.371920351148152e+00,...
       6.333131897833198e+00,...
       7.335666920593883e+00,...
       8.373706635544712e+00,...
       9.442353297358748e+00,...
       1.053748222747535e+01,...
       1.165561350236195e+01,...
       1.279380339874144e+01,...
       13];%13.736667272428120];
   %1.394955385079727e+01,... reduce to keep cn < 10
   
   s = 0;
   Tpowers{2} = T^2;
   d2 = norm(Tpowers{2}, 1)^(1/2);
   a1 = d2;
   if a1 <= th(1), m = 1; return; end
   Tpowers{4} = Tpowers{2}^2;
   d4 = norm(Tpowers{4}, 1)^(1/4);
   d6 = normAm(T, 6)^(1/6);
   a2 = max(d4, d6);
   if a2 <= th(2), m = 2; return; end
   Tpowers{6} = Tpowers{4}*Tpowers{2};
   d6 = norm(Tpowers{6}, 1)^(1/6);
   a2 = max(d4, d6);
   if a2 <= th(3), m = 3; return; end
   if a2 <= th(4), m = 4; return; end
   d8 = normAm(T, 8)^(1/8);
   a3 = max(d6, d8);
   if a3 <= th(6), m = 6; return; end
   Tpowers{8} = Tpowers{4}^2;
   d8 = norm(Tpowers{8}, 1)^(1/8);
   a3 = max(d6, d8);
   if a3 <= th(8), m = 8; return; end
   if a3 <= th(10), m = 10; return; end
   if a3 <= 2*th(8), m = 8; s = 1; return; end
   d10 = normAm(T, 10)^(1/10);
   a4 = max(d8, d10);
   a34 = min(a3, a4);
   if a34 <= th(12), m = 12; return; end
   if a3 <= 2*th(10), m = 10; s = 1; return; end
   if a3 <= 4*th(8), m = 8; s = 2; return; end
   if a34 <= th(15), m = 15; return; end
   if a34 <= 2*th(12), m = 12; s = 1; return; end
   if a3 <= 4*th(10), m = 10; s = 2; return; end
   if a3 <= 8*th(8), m = 8; s = 3; return; end
   if a34 <= th(18), m = 18; return; end
   if a34 <= 2*th(15), m = 15; s = 1; return; end
   if a34 <= 4*th(12), m = 12; s = 2; return; end
   if a3 <= 8*th(10), m = 10; s = 3; return; end
   d12 = normAm(T, 12)^(1/12);
   a5 = max(d10, d12);
   a345 = min([a3, a4, a5]);
   if a345 <= th(21), m = 21; return; end
   % Else we need to scale
   s = ceil(log2(a345/th(21)));
   a3 = a3/2^s;
   a34 = a34/2^s;
   a345 = a345/2^s;
   if a34 <= th(15), m = 15; return; end
   if a34 <= 2*th(12), m = 12; s = s + 1; return; end
   if a3 <= 4*th(10), m = 10; s = s + 2; return; end
   if a3 <= 8*th(8), m = 8; s = s + 3; return; end
   if a34 <= th(18), m = 18; return; end
   if a34 <= 2*th(15), m = 15; s = s + 1; return; end
   if a34 <= 4*th(12), m = 12; s = s + 2; return; end
   if a3 <= 8*th(10), m = 10; s = s + 3;  return; end
   if a345 <= th(21), m = 21; return;
   else
       % Shouldn't have gotten to here, error has occured.
       error('cosm:NoParam', 'Could not find parameters, please check A.')
   end     
end % of param_selection


function [X,nprods] = rational_approx(Tpowers, m)
% RATIONAL_APPROX  Rational approximant to the matrix cosine.
%
% This function computes the rational approximant to the matrix cosine
% based upon the Pade approximant to the exponential of degree M.

    I = eye(length(Tpowers{2}));
    switch m
        case 1
            nprods=1;
            p = [1 -1/4]; q = [1 1/4];
            P = p(1)*I + p(2)*Tpowers{2};
            R = q(1)*I + q(2)*Tpowers{2};
        case 2
            nprods=2;
            p = [1 -5/12 1/144]; q = [1 1/12 1/144];
            P = p(1)*I + p(2)*Tpowers{2} + p(3)*Tpowers{4}; 
            R = q(1)*I + q(2)*Tpowers{2} + q(3)*Tpowers{4};
        case 3
            nprods=3;
            p = [1 -9/20 11/600 -1/14400]; q = [1 1/20 1/600 1/14400];
            P = p(1)*I + p(2)*Tpowers{2} + p(3)*Tpowers{4} + p(4)*Tpowers{6};
            R = q(1)*I + q(2)*Tpowers{2} + q(3)*Tpowers{4} + q(4)*Tpowers{6};
        case 4
            nprods=4;
            p = [1, -13/28, 289/11760, -19/70560, 1/2822400];
            q = [1, 1/28, 3/3920, 1/70560, 1/2822400];
            Tpowers{8} = Tpowers{4}^2;
            P = p(1)*I + p(2)*Tpowers{2} + p(3)*Tpowers{4} + p(4)*Tpowers{6} + p(5)*Tpowers{8};
            R = q(1)*I + q(2)*Tpowers{2} + q(3)*Tpowers{4} + q(4)*Tpowers{6} + q(5)*Tpowers{8};
        case 6
            nprods=5;
            p = [1, -21/44, 533/17424, -533/914760, 169/43908480, -41/5269017600,...
                 1/442597478400];
            q = [1, 1/44, 5/17424, 1/365904, 1/43908480, 1/5269017600, 1/442597478400];
            P = p(1)*I + p(2)*Tpowers{2} + p(3)*Tpowers{4} + p(4)*Tpowers{6} +...
                Tpowers{6}*(p(5)*Tpowers{2} + p(6)*Tpowers{4} + p(7)*Tpowers{6});
            R = q(1)*I + q(2)*Tpowers{2} + q(3)*Tpowers{4} + q(4)*Tpowers{6} +...
                Tpowers{6}*(q(5)*Tpowers{2} + q(6)*Tpowers{4} + q(7)*Tpowers{6});
        case 8
            nprods=6;
            p = [1, -29/60, 1567/46800, -791/1029600, 9991/1349187840,...
                -499/15567552000, 529/8904639744000, -71/1869974346240000,...
                1/269276305858560000];
            q = [1, 1/60, 7/46800, 1/1029600, 1/192741120, 1/40475635200,...
                1/8904639744000, 1/1869974346240000, 1/269276305858560000];
    
            P = p(1)*I + p(2)*Tpowers{2} + p(3)*Tpowers{4} + p(4)*Tpowers{6} +...
                p(5)*Tpowers{8} + Tpowers{8}*(p(6)*Tpowers{2} + ...
                p(7)*Tpowers{4} + p(8)*Tpowers{6} + p(9)*Tpowers{8});
            R = q(1)*I + q(2)*Tpowers{2} + q(3)*Tpowers{4} + q(4)*Tpowers{6} +...
                q(5)*Tpowers{8} + Tpowers{8}*(q(6)*Tpowers{2} +...
                q(7)*Tpowers{4} + q(8)*Tpowers{6} + q(9)*Tpowers{8});
        case 10
            nprods=7;
            p = [1, -37/76, 10363/294576, -87/98192, 30749/3038060480, -362377/6187227171840,...
                462401/2598635412172800, -631/2273805985651200, 11449/56754197401853952000,...
                -109/2043151106466742272000, 1/449493243422683299840000];
            q = [1, 1/76, 9/98192, 1/2209320, 7/3906077760, 7/1145782809600,...
                7/371233630310400, 1/18190447885209600, 1/6306021933539328000,...
                1/2043151106466742272000, 1/449493243422683299840000];
            Tpowers{10} = Tpowers{4}*Tpowers{6};
            P = p(1)*I + p(2)*Tpowers{2} + p(3)*Tpowers{4} + p(4)*Tpowers{6} +...
                p(5)*Tpowers{8} + p(6)*Tpowers{10} + Tpowers{10}*(p(7)*Tpowers{2} +...
                p(8)*Tpowers{4} + p(9)*Tpowers{6} + p(10)*Tpowers{8} + p(11)*Tpowers{10});
            R = q(1)*I + q(2)*Tpowers{2} + q(3)*Tpowers{4} + q(4)*Tpowers{6} +...
                q(5)*Tpowers{8} + q(6)*Tpowers{10} + Tpowers{10}*(q(7)*Tpowers{2} +...
                q(8)*Tpowers{4} + q(9)*Tpowers{6} + q(10)*Tpowers{8} + q(11)*Tpowers{10});
        case 12
            nprods=8;
             p = [1, -45/92, 6451/177744, -97939/101314080, 1172939/96451004160, -8903/108507379680,...
                 105169/332967021843456, -40133/56604393713387520, 7411631/8083107422271737856000,...
                 -377521/581983734403565125632000, 1/4475076773576048640000, -31/1075505941177788352167936000,...
                 1/1677789268237349829381980160000];
             q = [1, 1/92, 11/177744, 5/20262816, 5/6430066944, 1/482255020800, 1/204200554521600,...
                 1/94340656188979200, 1/46189185270124216320, 1/23279349376142605025280,...
                 1/11639674688071302512640000, 1/5377529705888941760839680000, 1/1677789268237349829381980160000];
             Tpowers{10} = Tpowers{4}*Tpowers{6}; Tpowers{12} = Tpowers{6}^2;
             P = p(1)*I + p(2)*Tpowers{2} + p(3)*Tpowers{4} + p(4)*Tpowers{6} +...
                 p(5)*Tpowers{8} + p(6)*Tpowers{10} + p(7)*Tpowers{12} +...
                 Tpowers{12}*(p(8)*Tpowers{2} + p(9)*Tpowers{4} + p(10)*Tpowers{6} +...
                 p(11)*Tpowers{8} + p(12)*Tpowers{10} + p(13)*Tpowers{12});
             R = q(1)*I + q(2)*Tpowers{2} + q(3)*Tpowers{4} + q(4)*Tpowers{6} +...
                 q(5)*Tpowers{8} + q(6)*Tpowers{10} + q(7)*Tpowers{12} +...
                 Tpowers{12}*(q(8)*Tpowers{2} + q(9)*Tpowers{4} + q(10)*Tpowers{6} +...
                 q(11)*Tpowers{8} + q(12)*Tpowers{10} + q(13)*Tpowers{12});
       case 15
           nprods=9;
            p = [1, -57/116, 6793/181656, -114317/108993600, 4540429/315863452800,...
                -2093419/18951807168000, 61055671/118827830943360000,...
                -403985837/267384224682731520000, 1940595983/676482088447310745600000,...
                -17185433/4870671036820637368320000, 119069/42976509148417388544000000,...
                -1807879/1350150011406680678498304000000, 57191/153917101300361597348806656000000,...
                -6241/120055339014282045932069191680000000,...
                239/85719512056197380795497402859520000000,...
                -1/41145365786974742781838753372569600000000];
            q = [1, 1/116, 7/181656, 13/108993600, 13/45123350400, 11/18951807168000,...
                11/10802530085760000, 11/6856005761095680000, 11/4730643975156019200000,...
                1/316277340053288140800000, 1/243533551841031868416000000,...
                1/192878573058097239785472000000, 1/153917101300361597348806656000000,...
                1/120055339014282045932069191680000000,...
                1/85719512056197380795497402859520000000,...
                1/41145365786974742781838753372569600000000];
            Tpowers{10} = Tpowers{4}*Tpowers{6};
            P = p(1)*I + p(2)*Tpowers{2} + p(3)*Tpowers{4} + p(4)*Tpowers{6} +...
                p(5)*Tpowers{8} + p(6)*Tpowers{10} + Tpowers{10}*(p(7)*Tpowers{2} +...
                p(8)*Tpowers{4} + p(9)*Tpowers{6} + p(10)*Tpowers{8} + p(11)*Tpowers{10} +...
                Tpowers{10}*(p(12)*Tpowers{2} + p(13)*Tpowers{4} + p(14)*Tpowers{6} +...
                p(15)*Tpowers{8} + p(16)*Tpowers{10}));
            R = q(1)*I + q(2)*Tpowers{2} + q(3)*Tpowers{4} + q(4)*Tpowers{6} +...
                q(5)*Tpowers{8} + q(6)*Tpowers{10} + Tpowers{10}*(q(7)*Tpowers{2} +...
                q(8)*Tpowers{4} + q(9)*Tpowers{6} + q(10)*Tpowers{8} + q(11)*Tpowers{10} +...
                Tpowers{10}*(q(12)*Tpowers{2} + q(13)*Tpowers{4} + q(14)*Tpowers{6} +...
                q(15)*Tpowers{8} + q(16)*Tpowers{10}));
        case 18
            nprods=10;
            p = [1, -69/140, 8219/215600, -671/607600, 2193/137552800, -21920323/165789638784000,...
                134249447/196234645178880000, -3751133521/1613539369983340800000,...
                24063924851/4492093606033620787200000, -3117272252947/365638451156712597594931200000,...
                2278348165637/241321377763430314412654592000000, -181116301/24964280458285894594412544000000,...
                118410671/31027034283869611853055590400000000, -12439657/9300809806505855411951252275200000000,...
                3548761/11901654439670583707147802456883200000000, -211369/5400375702000527357118315364810752000000000,...
                12769/4838736628992472511978010566870433792000000000, -1/14473640356517073202984078528468746240000000000,...
                1/3375889771315468222156818412294164248002560000000000];
            q = [1, 1/140, 17/646800, 1/15038100, 1/7675446240, 1/4736846822400, 13/44052675448320000,...
                13/35462403735897600000, 13/31413242000235110400000, 13/30081320539425141719040000,...
                13/30682946950213644553420800000, 1/2531343123392625675657216000000, 1/2820639480351782895732326400000000,...
                1/3226811565522439632717781401600000000, 1/3740519966753612022246452200734720000000,...
                1/4320300561600421885694652291848601600000000, 1/4838736628992472511978010566870433792000000000,...
                1/4935511361572321962217570778207842467840000000000, 1/3375889771315468222156818412294164248002560000000000];
            Tpowers{10} = Tpowers{4}*Tpowers{6}; Tpowers{12} = Tpowers{6}^2;
           
            P = p(1)*I + p(2)*Tpowers{2} + p(3)*Tpowers{4} + p(4)*Tpowers{6} +...
                p(5)*Tpowers{8} + p(6)*Tpowers{10} + p(7)*Tpowers{12} + ...
                Tpowers{12}*(p(8)*Tpowers{2} + p(9)*Tpowers{4} + p(10)*Tpowers{6} + ...
                p(11)*Tpowers{8} + p(12)*Tpowers{10} + p(13)*Tpowers{12} + ...
                Tpowers{12}*(p(14)*Tpowers{2} + p(15)*Tpowers{4} + p(16)*Tpowers{6} + ...
                p(17)*Tpowers{8} + p(18)*Tpowers{10} + p(19)*Tpowers{12}));
            R = q(1)*I + q(2)*Tpowers{2} + q(3)*Tpowers{4} + q(4)*Tpowers{6} +...
                q(5)*Tpowers{8} + q(6)*Tpowers{10} + q(7)*Tpowers{12} + ...
                Tpowers{12}*(q(8)*Tpowers{2} + q(9)*Tpowers{4} + q(10)*Tpowers{6} + ...
                q(11)*Tpowers{8} + q(12)*Tpowers{10} + q(13)*Tpowers{12} + ...
                Tpowers{12}*(q(14)*Tpowers{2} + q(15)*Tpowers{4} + q(16)*Tpowers{6} + ...
                q(17)*Tpowers{8} + q(18)*Tpowers{10} + q(19)*Tpowers{12}));
        case 21
            nprods=11;
            p = [1, -81/164, 2533/65559, -2664719/2328655680, 14500309/847630667520,...
                -2750939/18442952985600, 4772125081/5775060311333913600,...
                -7205555041/2344674486401568921600, 6718682653/844082815104564811776000,...
                -175147477/11902734627741789511680000, 7557092027/383208908708499621408460800000,...
                -511087362307/26455098062894992010006770483200000,...
                1118382752029/80698631131054883627324652681953280000,...
                -928111901/128093065287388704170356591558656000000,...
                40219969879/14743639907643727238712214044992864256000000,...
                -37692173/52036376144624919666043108394092462080000000,...
                6811163/51897612474905919880266993438374882181120000000,...
                -19097941/1238692214551054495702212599387131687898972160000000,...
                131/121026069349746329449812955585281166776729600000000,...
                -1/25335803850102306486488931290809460129267712000000000,...
                461/818458448611321951708262761769450939989118460887040000000000,...
                -1/756255606516861483378434791874972668549945457859624960000000000];
            q = [1, 1/164, 5/262236, 19/465731136, 19/282543555840, 17/186478746854400,...
                17/160418341981497600, 17/156311632426771261440, 17/168816563020912962355200,...
                1/11687300516832435855360000, 1/14784435153793031357030400000,...
                1/19870280846697834143848857600000, 1/28088629004892058345744745103360000,...
                1/41383913400540965962730591118950400000, 1/63007008152323620678257324978601984000000,...
                1/98290932717624848258081426966619095040000000,...
                1/155692837424717759640800980315124646543360000000,...
                1/247738442910210899140442519877426337579794432000000,...
                1/390188047583582166146196968806946481688176230400000000,...
                1/593085832327044892542219392586558652166027870208000000000,...
                1/818458448611321951708262761769450939989118460887040000000000,...
                1/756255606516861483378434791874972668549945457859624960000000000];
            Tpowers{10} = Tpowers{4}*Tpowers{6}; Tpowers{12} = Tpowers{6}^2;
            Tpowers{14} = Tpowers{6}*Tpowers{8};
            
            P = p(1)*I + p(2)*Tpowers{2} + p(3)*Tpowers{4} + p(4)*Tpowers{6} +...
                p(5)*Tpowers{8} + p(6)*Tpowers{10} + p(7)*Tpowers{12} +...
                p(8)*Tpowers{14} + Tpowers{14}*(p(9)*Tpowers{2} + p(10)*Tpowers{4} +...
                p(11)*Tpowers{6} + p(12)*Tpowers{8} + p(13)*Tpowers{10} +...
                p(14)*Tpowers{12} + p(15)*Tpowers{14} + Tpowers{14}*(p(16)*Tpowers{2} +...
                p(17)*Tpowers{4} + p(18)*Tpowers{6} + p(19)*Tpowers{8} +...
                p(20)*Tpowers{10} + p(21)*Tpowers{12} + p(22)*Tpowers{14}));
            R = q(1)*I + q(2)*Tpowers{2} + q(3)*Tpowers{4} + q(4)*Tpowers{6} +...
                q(5)*Tpowers{8} + q(6)*Tpowers{10} + q(7)*Tpowers{12} +...
                q(8)*Tpowers{14} + Tpowers{14}*(q(9)*Tpowers{2} + q(10)*Tpowers{4} +...
                q(11)*Tpowers{6} + q(12)*Tpowers{8} + q(13)*Tpowers{10} +...
                q(14)*Tpowers{12} + q(15)*Tpowers{14} + Tpowers{14}*(q(16)*Tpowers{2} +...
                q(17)*Tpowers{4} + q(18)*Tpowers{6} + q(19)*Tpowers{8} +...
                q(20)*Tpowers{10} + q(21)*Tpowers{12} + q(22)*Tpowers{14}));
    end
    X = R\P;
    nprods=nprods+1;

end % of rational_approx


function Y = recompute_tbt(pos, depth, T)
% RECOMPUTE_TBT Recomputes a 2x2 diagonal block of the matrix.
%
% This function is called by recompute_diag to recompute the 2x2 diagonal
% blocks of a (quasi-)triangular matrix cosine to high accuracy.
% This generally increases the accuracy of the overall computation for
% nonnormal matrices.

    Y = 2^(-depth) .* T(pos:pos+1, pos:pos+1);
    if Y(2,1) == 0
        % Upper-tri block
        if Y(1,1) ~= Y(2,2)
            evalp = (Y(1,1) + Y(2,2))/2; evalm = (Y(1,1) - Y(2,2))/2;
            Y = [cos(Y(1,1))  -Y(1,2)*(sin(evalp)*sin(evalm))/evalm
                 0            cos(Y(2,2))];
        else
            Y = [cos(Y(1,1)) -Y(1,2)*sin(Y(1,1))
                 0           cos(Y(2,2))];
        end
    else
        % 2x2 block
        th = sqrt(-Y(1,2)*Y(2,1));
        Y = [cos(Y(1,1))*cosh(th)         -Y(1,2)*sin(Y(1,1))*sinh(th)/th
             -Y(2,1)*sin(Y(1,1))*sinh(th)/th   cos(Y(2,2))*cosh(th)];
    end
end % of recompute_tbt


function C = recompute_diag(C, T, depth)
% RECOMPUTE_DIAG Recompute the 2x2 diagonal blocks of C to high accuracy.
%
% This function uses an explicit formula for the matrix cosine of a 2x2
% matrix to accurately form the diagonal blocks of the (quasi-)triangular
% matrix C. This generally increases the accuracy of the overall
% computation for nonnormal matrices.

j = 1;
n = length(T);
    while j <= n-1
        if T(j+1,j) ~= 0
            C(j:j+1,j:j+1) = recompute_tbt(j, depth, T);
            j = j + 2; % Ignore next part of Schur block
        elseif j <= n-2 && T(j+2, j+1) ~= 0
            
            j = j + 1; % Next block is Schur block so leave it
        else
            C(j:j+1,j:j+1) = recompute_tbt(j, depth, T);
            j = j + 1; % On upper tri block so move to next bit
        end
    end
end % of recompute_diag
