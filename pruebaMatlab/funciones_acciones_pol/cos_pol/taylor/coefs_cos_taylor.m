function p = coefs_cos_taylor(m)
% p = coefs_cos_taylor(m)
% Coeficientes de la aproximaci�n de Taylor en el c�lculo de cos(A).
%
% Datos de entrada:
% - m: Orden de la aproximaci�n (1, 2, 3, 4, 5, 6, ...), de modo que el 
%      grado del polinomio a emplear ser� 2*m.
%
% Datos de salida:
% - p: Vector de m+1 componentes con los coeficientes de menor a mayor
%      grado. En realidad, del polinomio de grado 2*m, s�lo se devuelven 
%      los coeficientes situados en las posiciones pares (0, 2, ...,2*m).

if m<=42
    %Valores originales
    %p=vpa([1 -1/2, 1/24, -1/720, 1/40320, -1/3628800, 1/479001600, -1/87178291200, 1/20922789888000, -1/6402373705728000, 1/2432902008176640000, -1/1124000727777607680000, 1/620448401733239439360000, -1/403291461126605635584000000, 1/304888344611713860501504000000, -1/265252859812191058636308480000000, 1/263130836933693530167218012160000000, -1/295232799039604140847618609643520000000, 1/371993326789901217467999448150835200000000, -1/523022617466601111760007224100074291200000000, 1/815915283247897734345611269596115894272000000000, -1/1405006117752879898543142606244511569936384000000000, 1/2658271574788448768043625811014615890319638528000000000, -1/5502622159812088949850305428800254892961651752960000000000, 1/12413915592536072670862289047373375038521486354677760000000000, -1/30414093201713378043612608166064768844377641568960512000000000000, 1/80658175170943878571660636856403766975289505440883277824000000000000, -1/230843697339241380472092742683027581083278564571807941132288000000000000, 1/710998587804863451854045647463724949736497978881168458687447040000000000000, -1/2350561331282878571829474910515074683828862318181142924420699914240000000000000, 1/8320987112741390144276341183223364380754172606361245952449277696409600000000000000, -1/31469973260387937525653122354950764088012280797258232192163168247821107200000000000000, 1/126886932185884164103433389335161480802865516174545192198801894375214704230400000000000000, -1/544344939077443064003729240247842752644293064388798874532860126869671081148416000000000000000, 1/2480035542436830599600990418569171581047399201355367672371710738018221445712183296000000000000000, -1/11978571669969891796072783721689098736458938142546425857555362864628009582789845319680000000000000000, 1/61234458376886086861524070385274672740778091784697328983823014963978384987221689274204160000000000000000, -1/330788544151938641225953028221253782145683251820934971170611926835411235700971565459250872320000000000000000, 1/1885494701666050254987932260861146558230394535379329335672487982961844043495537923117729972224000000000000000000, -1/11324281178206297831457521158732046228731749579488251990048962825668835325234200766245086213177344000000000000000000, 1/71569457046263802294811533723186532165584657342365752577109445058227039255480148842668944867280814080000000000000000000, -1/475364333701284174842138206989404946643813294067993328617160934076743994734899148613007131808479167119360000000000000000000, 1/3314240134565353266999387579130131288000666286242049487118846032383059131291716864129885722968716753156177920000000000000000000]);    
    %Valores generados de nuevo por m� (Jos� M)
    p=vpa([1, -1/2, 1/24, -1/720, 1/40320, -1/3628800, 1/479001600, -1/87178291200, 1/20922789888000, -1/6402373705728000, 1/2432902008176640000, -1/1124000727777607680000, 1/620448401733239409999872, -1/403291461126605650322784256, 1/304888344611713836734530715648, -1/265252859812191032188804700045312, 1/263130836933693517766352317727113216, -1/295232799039604119555149671006000381952, 1/371993326789901177492420297158468206329856, -1/523022617466601037913697377988137380787257344, 1/815915283247897683795548521301193790359984930816, -1/1405006117752879788779635797590784832178972610527232, 1/2658271574788448529134213028096241889243150262529425408, -1/5502622159812088456668950435842974564586819473162983440384, 1/12413915592536072528327568319343857274511609591659416151654400, -1/30414093201713375576366966406747986832057064836514787179557289984, 1/80658175170943876845634591553351679477960544579306048386139594686464, -1/230843697339241379243718839060267085502544784965628964557765331531071488, 1/710998587804863481025438135085696633485732409534385895375283304551881375744, -1/2350561331282878906297796280456247634956966273955390268712005058924708557225984, 1/8320987112741391580056396102959641077457945541076708813599085350531187384917164032, -1/31469973260387939390320343330721249710233204778005956144519390914718240063804258910208, 1/126886932185884165437806897585122925290119029064705209778508545103477590511637067401789440, -1/544344939077443069445496060275635856761283034568718387417404234993819829995466026946857533440, 1/2480035542436830547970901153987107983847555399761061789915503815309070879417337773547217359994880, -1/11978571669969890269925854460558840225267029209529303278944419871214396524861374498691473966836482048, 1/61234458376886076682034243918084408426143679367126656631657903381829221022872956916891969827292894461952, -1/330788544151938558975078458606627397928594841525087028177611631961228972749086355889619432768006702663467008, 1/1885494701666049846649767567286674986020753759697889931196791720482648043560619012598537844549685032003930423296, -1/11324281178206294606285193764734547659641544873910049469239570110699644621282776159978832493218689331409071173009408, 1/71569457046263778832073404098641551692451427821500630228331524401978643519022131505852398484420816675798776564959674368, -1/475364333701283981804950871934204857403260987909684614932289567004882674634326008655216234173410083475042065689611178344448, 1/3314240134565351991893962785187002255986138585985099085000359647021178112607661449751964466234594461331925608329126314254532608]);
    p=double(p);
    p=p(1:m+1);
else
    p=vpa(coefs_cos_taylor_sym(m));
    p=double(p);
end

end
