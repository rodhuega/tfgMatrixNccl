% Script to compile psacore.c and psacore_hqr.c for different
% architectures, depending on the MATLAB version and architecture.

% Get the MATLAB version
  v = ver('matlab');
  m_ver = str2num(v.Version);
  c = computer;

% Use the correct compile command for the particular machine
  if m_ver < 6, 
    switch c
    case 'PCWIN'   % NT Lab
      disp('Need ztrsm for function calls')
      mex -O psacore.c zbandqrf.c "c:\matlabr12\extern\lib\win32\microsoft\msvc60\libmwlapack.lib"
      break
    case 'SOL2'    % beech.comlab
      disp('Uses ztrsm_ for function calls')
      mex -O psacore.c zbandqrf.c ztrsm.f dswap.f zgemv.f ztrmv.f zgerc.f ztrmm.f zgemm.f dznrm2.f zdscal.f zscal.f zcopy.f lapack_solaris.a
      break
    case 'HP700'   % l9.nag
      disp('Need ztrsm for function calls')
      mex -O FC=f90 psacore.c zbandqrf.c  dswap.f /usr/lib/liblapack.a
      break
    case 'SGI64'   % l12.nag
      disp('Uses ztrsm_ for function calls')
      mex -O psacore.c zbandqrf.c ztrsm.f dswap.f
      break
    case 'IBM_RS'  % l21.nag, l13.nag
      disp('Uses ztrsm for function calls')
      mex -O psacore.c zbandqrf.c ztrsm.f dswap.f zgemv.f ztrmv.f zgerc.f ztrmm.f zgemm.f dznrm2.f zdscal.f zscal.f zcopy.f lapack_rs6k.a -L/users/systems/pub/MATLAB/matlab6/bin/ibm_rs -L/users/systems/pub/MATLAB/matlab5/extern/lib/ibm_rs
      break
    case 'ALPHA'   % es40.nag
      disp('Need ztrsm_ for function calls')
      mex -O psacore.c zbandqrf.c ztrsm.f dswap.f zgemv.f ztrmv.f zgerc.f ztrmm.f zgemm.f dznrm2.f zdscal.f zscal.f zcopy.f lapack_alpha.a
      break
    case 'LNX86'   % clpc62.comlab
      disp('Uses ztrsm_ for function calls')
      mex -O psacore.c zbandqrf.c
      break
    otherwise
      disp('Unknown computer type');
      break
    end;
  else
    switch c
    case 'PCWIN'   % NT Lab
      disp('Need ztrsm for function calls')
      mex -O psacore.c zbandqrf.c "d:\matlabr12\extern\lib\win32\microsoft\msvc60\libmwlapack.lib"
      break
    case 'SOL2'    % beech.comlab
      disp('Uses ztrsm_ for function calls')
      mex -O psacore.c zbandqrf.c
      break
    case 'HPUX'    % l9.nag
      disp('Uses ztrsm for function calls')
      mex -O psacore.c zbandqrf.c
      break
    case 'HP700'   % l4.nag
      disp('Need ztrsm for function calls')
      mex -O psacore.c zbandqrf.c
      break
    case 'SGI'     % origin3400.nag (not working, no MATLAB binary)
      disp('Uses ztrsm_ for function calls')
      mex -O psacore.c zbandqrf.c
      break
    case 'IBM_RS'  % l21.nag
      disp('Uses ztrsm for function calls')
      mex -O psacore.c zbandqrf.c -L/users/systems/pub/MATLAB/matlab6/bin/ibm_rs -lmwlapack
      break
    case 'ALPHA'   % es40.nag
      disp('Need ztrsm_ for function calls')
      mex -O psacore.c zbandqrf.c
      break
    case 'GLNX86'  % henrici.comlab
      disp('Uses ztrsm_ for function calls')
      mex -O psacore.c zbandqrf.c
      break
    otherwise
      disp('Unknown computer type');
      break
    end;

  end;
