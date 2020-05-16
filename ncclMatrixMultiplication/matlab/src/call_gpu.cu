
#include <mex.h>
#include <string.h>
#include "cuda.h"
#include "cublas_v2.h"
#include "../../include/cpp/OperationType.h"
#include "../../include/cuda/ErrorCheckingCuda.cuh"
#include "../../include/cuda/MatrixMain.cuh"
#include "../../include/cuda/NcclMultiplicationEnvironment.cuh"
#include <vector>

using namespace std;

static int initiated = 0;

enum type_method_f{ no_method, taylor, bernoulli, hermite };

enum eval_method{ PatMey };

class funcion_matricial {
  protected:
    int n; /* Matrix size */
    NcclMultiplicationEnvironment<double> *ncclMultEnv ;
    vector< MatrixMain<double>* > pA;
    MatrixMain<double>* R; /* Matrix result */
    int scaled;
    int evaluated;
    int unscaled;
    int nProd;
    eval_method e_method;
    type_method_f metodo_f;

  public: 
    funcion_matricial( int n, type_method_f metodo_f, eval_method e_method, const double *A );
    ~funcion_matricial( );

    int getN( ) const { return n; }
    int getQ( ) const { return pA.size()-1; } /* Returns the order of the PatMey polynomial */
    void get( const int i, double *A );
    virtual void power( );
    //void power( const int i );
    double norm1( const int i );
    void scale( const int s, const double e );
    virtual void scale( const int s ) = 0;
    int evaluate( const int m, const double *p );
    int eval_PatMey( const int m, const double *p );
    virtual void unscale( const int s ) = 0;
    void free( const int n );
    void finalize( mxArray **plhs );
};

class cos_matricial : public funcion_matricial {
  public: 
    cos_matricial( int n, type_method_f metodo_f, eval_method e_method, const double *A );

    void power( );
    void scale( const int s );
    void unscale( const int s );

};

class cosh_matricial : public funcion_matricial {
  public: 
    cosh_matricial( int n, type_method_f metodo_f, eval_method e_method, const double *A );

    void power( );
    void scale( const int s );
    void unscale( const int s );

};

class exp_matricial : public funcion_matricial {
  public: 
    exp_matricial( int n, type_method_f metodo_f, eval_method e_method, const double *A );

    void power( );
    void scale( const int s );
    void unscale( const int s );

};

funcion_matricial::funcion_matricial( int n, type_method_f metodo_f, eval_method e_method, const double *A ) : n(n), scaled(0), evaluated(0), unscaled(0), metodo_f(metodo_f), e_method(e_method) {
  //Crear el entorno multiplicativo aqui?
  //Crear R aqui?
  int gpuSizeSystem;
  CUDACHECK(cudaGetDeviceCount(&gpuSizeSystem));
  ncclMultEnv = new NcclMultiplicationEnvironment<double>(gpuSizeSystem, 0, MultDouble, false);
  R = new MatrixMain<double>(ncclMultEnv, n, n);
  MatrixMain<double> MA = MatrixMain<double>(ncclMultEnv, n, n, (double*)A);

  pA.push_back( new MatrixMain<double>(ncclMultEnv, n, n, (double*)A) );
  nProd = 0;
}

cos_matricial::cos_matricial( int n, type_method_f metodo_f, eval_method e_method, const double *A ) : funcion_matricial(n,metodo_f,e_method,A) { 
}

cosh_matricial::cosh_matricial( int n, type_method_f metodo_f, eval_method e_method, const double *A ) : funcion_matricial(n,metodo_f,e_method,A) { 
}

exp_matricial::exp_matricial( int n, type_method_f metodo_f, eval_method e_method, const double *A ) : funcion_matricial(n,metodo_f,e_method,A) { 
}

void funcion_matricial::power( ) {
  MatrixMain<double> *auxMult;
  auxMult=&((*pA[0]) * (*pA[pA.size()-1]));
  pA.push_back(auxMult  );
  nProd++;
}

static bool a = true;

void cos_matricial::power( ) {
  funcion_matricial::power();
  if( a && metodo_f != bernoulli ) {
    pA[0] = pA[1];
    pA.pop_back();
    a = false;
  }
}

void cosh_matricial::power( ) {
  funcion_matricial::power();
  if( a && metodo_f != bernoulli ) {
    pA[0] = pA[1];
    pA.pop_back();
    a = false;
  }
}

void exp_matricial::power( ) {
  funcion_matricial::power();
}

void funcion_matricial::get( int i, double *A ) {
  if( !( i>=0 && i<pA.size() ) ) return;
  pA[i]->getHostMatrixInThisPointer(A);
}

double funcion_matricial::norm1( const int i ) {
  if( i < 0 || i > pA.size() ) {
    printf("There's no MatrixMain %d\n",i);
    return 0.0;
  }
  return pA[i]->norm1();
}

void funcion_matricial::free( int n ) {
  if( !( n>0 && n<pA.size() ) ) return;
  for(auto it=pA.end()-n;it<pA.end();it++)
  {
    (*it)->setDeleteMatrixHostAtDestroyment(true);
    delete *it;
  }
  pA.erase(pA.end()-n, pA.end());
}

void funcion_matricial::scale( const int s, const double e ) {
  if( scaled ) return;
  int i = 1;
  for( auto it : pA ) 
  {
    (*it)/=pow( e, s*(i++));
  }
  scaled = 1;

}

void cos_matricial::scale( const int s ) {
  double e;
  if( metodo_f == bernoulli ) e = 2.0;
  else e = 4.0;
  funcion_matricial::scale( s, e );
}

void cosh_matricial::scale( const int s ) {
  double e;
  if( metodo_f == bernoulli ) e = 2.0;
  else e = 4.0;
  funcion_matricial::scale( s, e );
}

void exp_matricial::scale( const int s ) {
  funcion_matricial::scale( s, 2.0 );
}

int funcion_matricial::evaluate( const int m, const double *p ) {
  if( evaluated ) return 0;
  int nProd;
  switch( e_method ) {
    case PatMey:
          nProd = eval_PatMey( m, p );
          break;
    default: cout << "No valid evaluation method " << endl;
  }
  evaluated = 1;
  return nProd;
}

int funcion_matricial::eval_PatMey( const int m, const double *p ) {
  int n = pA[0]->getRowsReal();
  int degree = m - 1;
  int q = pA.size();
  int c = degree + 1;
  int k = degree / q;
  int nProd = 0;

  R->setMatrixHostToFullValue( 0.0 ); /* R=zeros(n); */
  for( int j = k; j > 0; j-- ) {
    int inic;
    if( j == k ) {
        inic = q;
    } else {
        inic = q-1;
    }
    for( int i = inic; i > 0; i-- ) {
        // axpy( p[c-1], pA[i-1], *R ); /* R += p[c] * pA[i]; */
        R->axpy(p[c-1],*pA[i-1]);
        c = c - 1;
    }
    /* R = R + p[c] * I; */
    *R += p[c-1];
    c = c - 1;
    if( j != 1 ) {
        *R *= *pA[q-1]; /* R = R * pA[q]; */
        nProd = nProd + 1;
    }
  }
  return nProd;
}

void cos_matricial::unscale( const int s ) {
  if( unscaled ) return;
  for( int i=0; i<s; i++ ) {
    /* F:=2*F*F-I; */
    *R = 2.0*(*R)*(*R)-1.0;
  }
  unscaled = 1;
}

void cosh_matricial::unscale( const int s ) {
  if( unscaled ) return;
  for( int i=0; i<s; i++ ) {
    /* F:=2*F*F-I; */
    *R = 2.0*(*R)*(*R)-1.0;
  }
  unscaled = 1;
}

void exp_matricial::unscale( const int s ) {
  if( unscaled ) return;
  for( int i=0; i<s; i++ ) {
    /* F:=F*F; */
    *R = (*R)*(*R);
  }
  unscaled = 1;
}

void funcion_matricial::finalize( mxArray **plhs ) {
  *plhs = mxCreateDoubleMatrix((mwSize)n, (mwSize)n, mxREAL);
  R->getHostMatrixInThisPointer( mxGetPr(*plhs) );
}

funcion_matricial::~funcion_matricial() {
  int i;
  for(i =0;i<pA.size();i++)
  {
    pA[i]->setDeleteMatrixHostAtDestroyment(true);
    delete pA[i];
  }
  pA.clear();
  R->setDeleteMatrixHostAtDestroyment(true);
  delete R;
  delete ncclMultEnv;
}

funcion_matricial *F;

/* Interface routines */
void initialize() {
  /* Check number of GPUs */
  int deviceCount;
  CUDACHECK(cudaGetDeviceCount(&deviceCount));
  if( deviceCount<1 ) {
    mexErrMsgIdAndTxt("MATLAB:call_gpu","Not enough GPUs available.");
  }
  /*
  if( deviceCount>2 ) {
    mexPrintf("MATLAB:Warning: Not yet implemented for more than 1 GPU.\n");
  }
  */
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {

  /* Check for proper number of arguments. */
  if(nlhs>1) {
    mexErrMsgIdAndTxt("MATLAB:call_gpu:maxlhs","Too many output arguments.");
  }

  char comando[80];
  mxGetString( prhs[0], comando, 80 );
  if( strcmp( comando, "init" ) && !initiated ) {
    mexErrMsgIdAndTxt("MATLAB:call_gpu:invalidCommand","Not yet initiated.");
  }
  if( !strcmp( comando, "init" ) ) {
    if( initiated ) return;
    if( nrhs!=4 ) {
      mexErrMsgIdAndTxt("MATLAB:call_gpu:invalidNumInputs","Arguments: cos|exp|cosh, taylor|bernoulli|hermite, matrix ");
    } 
    char funcion[80];
    mxGetString( prhs[1], funcion, 80 );
    if( !strcmp( funcion, "cos" ) && !strcmp( funcion, "exp" ) && !strcmp( funcion, "cosh" ) ) {
      mexErrMsgIdAndTxt("MATLAB:call_gpu:invalidInput","Function should be exp or cosh.");
    }
    char pol_method_name[80];
    mxGetString( prhs[2], pol_method_name, 80 );
    type_method_f metodo_f = no_method;
    if( !strcmp( pol_method_name, "taylor" ) )    metodo_f = taylor;
    if( !strcmp( pol_method_name, "bernoulli" ) ) metodo_f = bernoulli;
    if( !strcmp( pol_method_name, "hermite" ) )   metodo_f = hermite;
    if( metodo_f == no_method ) {
      mexErrMsgIdAndTxt("MATLAB:call_gpu:invalidInput","Evaluation method should be taylor, bernoulli or hermite.");
    } 
    eval_method e_method = PatMey;
    if( !mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) ) {
      mexErrMsgIdAndTxt("MATLAB:call_gpu:invalidInput","Matrix must be real.");
    }
    if( mxGetM(prhs[3]) != mxGetN(prhs[3]) ) {
      mexErrMsgIdAndTxt("MATLAB:call_gpu:invalidInput","Matrix must be square.");
    }
    initialize();
    if( !strcmp( funcion, "cos" ) ) {
      F = new cos_matricial( mxGetM(prhs[3]), metodo_f, e_method, mxGetPr(prhs[3]) );
    } 
    if( !strcmp( funcion, "cosh" ) ) {
      F = new cosh_matricial( mxGetM(prhs[3]), metodo_f, e_method, mxGetPr(prhs[3]) );
    } 
    if( !strcmp( funcion, "exp" ) ) {
      F = new exp_matricial( mxGetM(prhs[3]), metodo_f, e_method, mxGetPr(prhs[3]) );
    } 
    initiated = 1;
  } else if( !strcmp( comando, "power" ) ) {
    F->power( );
    if( nlhs==1 ) {
      F->get( F->getQ(), mxGetPr( plhs[0] = mxCreateDoubleMatrix((mwSize)F->getN(), (mwSize)F->getN(), mxREAL) ) );
    }
  } else if( !strcmp( comando, "norm1" ) ) {
    if( nrhs!=2 ) {
      mexErrMsgIdAndTxt("MATLAB:call_gpu:invalidNumInputs","A matrix index (integer) as second argument is required.");
    } 
    *mxGetPr( plhs[0] = mxCreateDoubleMatrix((mwSize)1, (mwSize)1, mxREAL) ) = F->norm1((int) *mxGetPr(prhs[1])-1);
  } else if( !strcmp( comando, "scale" ) ) {
    if( nrhs!=2 ) {
      mexErrMsgIdAndTxt("MATLAB:call_gpu:invalidNumInputs","An integer with the scaling as second argument is required.");
    } 
    F->scale((int) *mxGetPr(prhs[1]));
  } else if( !strcmp( comando, "evaluate" ) ) { 
    if( nrhs!=2 ) {
      mexErrMsgIdAndTxt("MATLAB:call_gpu:invalidNumInputs","An array of coefficients as second argument is required.");
    } 
    *mxGetPr( plhs[0] = mxCreateDoubleMatrix((mwSize)1, (mwSize)1, mxREAL) ) = F->eval_PatMey( mxGetN(prhs[1]), mxGetPr(prhs[1]) );
  } else if( !strcmp( comando, "unscale" ) ) {
    if( nrhs!=2 ) {
      mexErrMsgIdAndTxt("MATLAB:call_gpu:invalidNumInputs","An integer with the scaling as second argument is required.");
    } 
    F->unscale((int) *mxGetPr(prhs[1]));
  } else if( !strcmp( comando, "free" ) ) {
    if( nrhs!=2 ) {
      mexErrMsgIdAndTxt("MATLAB:call_gpu:invalidNumInputs","An integer with the power as second argument is required.");
    } 
    F->free((int) *mxGetPr(prhs[1]));
  } else if( !strcmp( comando, "finalize" ) ) {
    F->finalize( &plhs[0] );
    if( F!=NULL ) {
      delete F;
      F = NULL;
    }
    initiated = 0;
  } else {
    printf("Command unknown\n");
  }
}

