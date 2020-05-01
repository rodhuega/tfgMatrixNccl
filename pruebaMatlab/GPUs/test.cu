
#include "Matrix.h"
#include "error_macros.h"
#include <iostream>
#include <cublas_v2.h>
#include <vector>

using namespace std;

int polyvalm_paterson_stockmeyer( const vector<double>& p, const vector< Matrix<double> >& pA, Matrix<double>& E ) {
  int n = pA[0].getN();
  int m = p.size() - 1;
  int q = pA.size();
  int c = m + 1;
  int k = m / q;
  cout << " n = " << n << endl;
  cout << " m = " << m << endl;
  cout << " q = " << q << endl;
  cout << " c = " << c << endl;
  cout << " k = " << k << endl;

  // E=zeros(n);
  E.set( 0.0 );
  int nProd = 0;
  for( int j = k; j > 0; j-- ) { 
    int inic;
    if( j == k ) {
        inic = q;
    } else {
        inic = q-1;
    }
    cout << "polyeval: j = " << j << " inic = " << inic << endl;
    for( int i = inic; i > 0; i-- ) {
        cout << "polyeval: i = " << i << endl;
        //E += p[c] * pA[i];
        axpy( p[c], pA[i-1], E );
        c = c - 1;
    }
    //E = E + p[c] * I;
    cout << "2. polyeval: j = " << j << endl;
    E = E + p[c-1];
    cout << "3. polyeval: j = " << j << endl;
    c = c - 1;
    if( j!=1 ) {
        //E = E * pA[q];
        E *= pA[q-1];
        nProd = nProd + 1;
    }
  }
  return nProd;
}

#define A(i,j)          A[ (i) + ((j)*(n)) ]

int main () {

   int n = 4; /* Tamanyo */
   int p = 16;
   int q = 4;

   double *A = (double *) malloc( n*n*sizeof(double) );
   for( int i=0; i<n; i++ ) {
     for( int j=0; j<n; j++ ) {
       A( i, j ) = 2.0 * ( (double) rand() / RAND_MAX ) - 1.0;
     }
   }
   vector<double> coef;
   for( int i=0; i<p; i++ ) {
     coef.push_back(2.0 * ( (double) rand() / RAND_MAX ) - 1.0);
   }

   cublasHandle_t handle;
   CUBLAS_SAFE_CALL( cublasCreate(&handle) );

   /* Generamos potencias de matrices */
   vector< Matrix<double> > pA;
   //Matrix<double> MA( n, A, handle );
   //pA.push_back( std::move(MA) );
   //pA.push_back( MA );
   pA.push_back( Matrix<double>( n, A, handle ) ); /* Se llama al constructor de movimiento */
   printf("n = %d, 0x%x\n",pA[0].getN(),pA.front().getA());
   cout << endl << "Entrando en el bucle" << endl;
   for( int i = 1; i<q; i++ ) {
     cout << "Computing power " << i << endl;
     //pA[i-1] * pA[0];
     //Matrix<double> MB = pA[i-1] * pA[0];
     //printf("n = %d, 0x%x\n",MB.getN(),MB.getA());
     //pA.push_back( MB );
     pA.push_back( pA[i-1] * pA[0] );
     //pA[0] = std::move( pA[i-1] * pA[0] );
     //pA[0] = pA[i-1] * pA[0];
     //printf("n = %d, 0x%x\n",pA[i].getN(),pA[i].getA());
   }
   cout << "Size = " << pA.size() << endl;
#ifdef NOCOMPILE

   Matrix<double> E(n,0.0,handle);
   int nProd = polyvalm_paterson_stockmeyer( coef, pA, E );
   cout << "Nprod = " << nProd << endl;
#endif

   CUBLAS_SAFE_CALL( cublasDestroy(handle) );
   free(A);
}


