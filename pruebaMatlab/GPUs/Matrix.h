
#ifndef MATRIX_GPU
#define MATRIX_GPU

#include <iostream>
#include "error_macros.h"
#include <cublas_v2.h>

template <typename T> class Matrix;
template <typename T> Matrix<T>  operator*( Matrix<T> A, const Matrix<T>& B );
template <typename T> Matrix<T>  operator+( Matrix<T> A, const Matrix<T>& B );
template <typename T> Matrix<T>  operator+( Matrix<T> A, const T& v );
template <typename T> Matrix<T>  operator-( Matrix<T> A, const T& v );
template <typename T> Matrix<T>  operator*( const T& v, Matrix<T> A );
template <typename T> Matrix<T>  operator*( Matrix<T> A, const T& v ); 
template <typename T> Matrix<T>& axpy( const T& a, const Matrix<T>& X, Matrix<T>& Y );
template <typename T> std::ostream& operator<<( std::ostream& os, Matrix<T>& A );


template <typename T>
class Matrix {
  private:
    int n; 
    T *d_A;
    cublasHandle_t handle;
  public:
    Matrix( );
    Matrix( const int n );
    Matrix( const int n, cublasHandle_t handle );
    Matrix( const int n, const T *A, cublasHandle_t handle );
    Matrix( const int n, const T v, cublasHandle_t handle );
    Matrix( const Matrix& A );
    Matrix( Matrix&& A );
    int getN( ) const { return n; }
    size_t getA( ) const { return size_t(d_A); }
    cublasHandle_t getHandle( ) const { return handle; }
    void setHandle( const cublasHandle_t h ) { handle = h; }
    void get( T *A );
    void set( const T &v );
    void put( const T *A ) const;
    ~Matrix( );
    T norm1() const;
    T operator()( const int i, const int j );
    Matrix& operator=( const Matrix& A );
    Matrix& operator=( Matrix&& A );
    Matrix& operator*=( const Matrix& A );
    friend Matrix<T> operator*<>( Matrix<T> A, const Matrix<T>& B );
    Matrix& operator+=( const Matrix& A );
    friend Matrix<T> operator+<>( Matrix<T> A, const Matrix<T>& B );
    Matrix& operator+=( const T& v );
    friend Matrix<T> operator+<>( Matrix<T> A, const T& v );
    Matrix& operator-=( const T& v );
    friend Matrix<T> operator-<>( Matrix<T> A, const T& v );
    Matrix& operator*=( const T& v );
    friend Matrix<T>  operator*<>( const T& v, Matrix<T> A );
    friend Matrix<T>  operator*<>( Matrix<T> A, const T& v ); 
    friend Matrix<T>& axpy<>( const T& a, const Matrix<T>& X, Matrix<T>& Y );
    void scal( const T& s );
    friend std::ostream& operator<<<>( std::ostream& os, Matrix<T>& A );

};

template <class T>
Matrix<T>::Matrix( ) {
  d_A = nullptr;
}

template <class T>
Matrix<T>::Matrix( const int n ) : n(n) {
  CUDA_SAFE_CALL( cudaMalloc( (void **) &d_A, n*n*sizeof(T) ) );
}

template <class T>
Matrix<T>::Matrix( const int n, const cublasHandle_t handle ) : Matrix( n ) {
  this->handle = handle;
}

template <class T>
Matrix<T>::Matrix( const int n, const T *A, cublasHandle_t handle ) : Matrix( n, handle ) {
  put( A );
}

template <class T>
Matrix<T>::Matrix( const int n, const T v, cublasHandle_t handle ) : Matrix( n, handle ) {
  set( v );
}

template <class T>
Matrix<T>::Matrix( const Matrix<T>& A ) : Matrix( A.n, A.handle ) {
  CUDA_SAFE_CALL( cudaMemcpy( d_A, A.d_A, n*n*sizeof(T), cudaMemcpyDeviceToDevice ) );
}

template <class T>
Matrix<T>::Matrix( Matrix<T>&& A ) : Matrix() {
  *this = std::move(A);
}

template <class T>
void Matrix<T>::scal( const T& s) {
  CUBLAS_SAFE_CALL( cublasDscal( handle, n*n, &s, this->d_A, 1 ) );
}

template <class T>
void Matrix<T>::set( const T& v ) {
  T *A = (T *) malloc(n*n*sizeof(T));
  for( size_t i = 0; i < n*n; i++ ) {
    A[i] = v;
  }
  CUDA_SAFE_CALL( cudaMemcpy( d_A, A, n*n*sizeof(T), cudaMemcpyHostToDevice ) );
  free(A);
}

template <class T>
void Matrix<T>::get( T *A ) {
  CUDA_SAFE_CALL( cudaMemcpy( A, d_A, n*n*sizeof(T), cudaMemcpyDeviceToHost ) );
}

template <class T>
void Matrix<T>::put( const T *A ) const {
  CUDA_SAFE_CALL( cudaMemcpy( d_A, A, n*n*sizeof(T), cudaMemcpyHostToDevice ) );
}

template <class T>
Matrix<T>::~Matrix( ) {
  if( d_A != NULL ) {
    CUDA_SAFE_CALL( cudaFree( d_A ) );
    d_A = NULL;
  }
}

template <class T>
T Matrix<T>::norm1( ) const {
  int inc = 1;
  double max = 0.0;
  for( int i=0; i<n; i++ ) {
    double a;
    cublasDasum( handle, n, &d_A[i*n], inc, &a );
    max = a>max ? a : max;
  }
  return max;
}

template <class T>
T Matrix<T>::operator()( const int i, const int j ) {
  T a;
  CUDA_SAFE_CALL( cudaMemcpy( &a, &d_A[i+j*n], sizeof(T), cudaMemcpyDeviceToHost ) );
  return a;
}

template <class T>
Matrix<T>& Matrix<T>::operator=( const Matrix<T>& A ) {
  if( this != &A ) {
    CUDA_SAFE_CALL( cudaMemcpy( d_A, A.d_A, n*n*sizeof(T), cudaMemcpyDeviceToDevice ) );
  }
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator=( Matrix<T>&& A ) {
  if( this != &A ) {
    n = std::move(A.n);
    handle = std::move(A.handle);
    if( d_A != NULL ) {
      CUDA_SAFE_CALL( cudaFree( d_A ) );
    }
    d_A = std::move(A.d_A);
    A.d_A = NULL;
  }
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator+=( const Matrix<T>& A ) {
  if( this != &A ) {
    const T ONE = 1.0;
    CUBLAS_SAFE_CALL( cublasDaxpy( handle, n*n, &ONE, A.d_A, 1, this->d_A, 1 ) );
  } else {
    const T TWO = 2.0;
    CUBLAS_SAFE_CALL( cublasDscal( handle, n*n, &TWO, this->d_A, 1 ) );
  }
  return *this;
}

template <class T> 
Matrix<T> operator+( Matrix<T> A, const Matrix<T>& B ) {
  A += B;
  return A;
}

template <class T>
Matrix<T>& Matrix<T>::operator+=( const T& v ) {
  T *d_V;
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_V, sizeof(T) ) );
  CUDA_SAFE_CALL( cudaMemcpy( d_V, &v, sizeof(T), cudaMemcpyHostToDevice ) );
  const T ONE = 1.0;
  CUBLAS_SAFE_CALL( cublasDaxpy( handle, n, &ONE, d_V, 0, d_A, n+1 ) );
  CUDA_SAFE_CALL( cudaFree( d_V ) );
  return *this;
}

template <class T> 
Matrix<T> operator+( Matrix<T> A, const T& v ) {
  A += v;
  return A;
}

template <class T>
Matrix<T>& Matrix<T>::operator-=( const T& v ) {
  T *d_V;
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_V, sizeof(T) ) );
  T w = -v;
  CUDA_SAFE_CALL( cudaMemcpy( d_V, &w, sizeof(T), cudaMemcpyHostToDevice ) );
  const T ONE = 1.0;
  CUBLAS_SAFE_CALL( cublasDaxpy( handle, n, &ONE, d_V, 0, d_A, n+1 ) );
  CUDA_SAFE_CALL( cudaFree( d_V ) );
  return *this;
}

template <class T> 
Matrix<T> operator-( Matrix<T> A, const T& v ) {
  A -= v;
  return A;
}

template <class T>
Matrix<T>& Matrix<T>::operator*=( const Matrix<T>& A ) {
  T *d_B;
  CUDA_SAFE_CALL( cudaMalloc( (void **) &d_B, n*n*sizeof(T) ) );
  CUDA_SAFE_CALL( cudaMemcpy( d_B, d_A, n*n*sizeof(T), cudaMemcpyDeviceToDevice ) );
  const T ZERO = 0.0;
  const T ONE = 1.0;
  if( this != &A ) {
    CUBLAS_SAFE_CALL( cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &ONE, d_B, n, A.d_A, n, &ZERO, d_A, n ) );
  } else {
    T *d_C;
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_C, n*n*sizeof(T) ) );
    CUDA_SAFE_CALL( cudaMemcpy( d_C, A.d_A, n*n*sizeof(T), cudaMemcpyDeviceToDevice ) );
    CUBLAS_SAFE_CALL( cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &ONE, d_B, n, d_C, n, &ZERO, d_A, n ) );
    CUDA_SAFE_CALL( cudaFree( d_C ) );
  }
  CUDA_SAFE_CALL( cudaFree( d_B ) );
  return *this;
}

template <class T> 
Matrix<T> operator*( Matrix<T> A, const Matrix<T>& B ) {
  A *= B;
  return A;
}

template <class T>
Matrix<T>& Matrix<T>::operator*=( const T& v ) {
  CUBLAS_SAFE_CALL( cublasDscal( handle, n*n, &v, this->d_A, 1 ) );
  return *this;
}

template <class T> 
Matrix<T> operator*( const T& v, Matrix<T> A ) {
  A *= v;
  return A;
}

template <class T> 
Matrix<T> operator*( Matrix<T> A, const T& v ) {
  A *= v;
  return A;
}

template <class T>
Matrix<T>& axpy( const T& a, const Matrix<T>& X, Matrix<T>& Y ) {
  cublasHandle_t handle = X.getHandle();
  CUBLAS_SAFE_CALL( cublasDaxpy( handle, X.getN()*X.getN(), &a, X.d_A, 1, Y.d_A, 1 ) );
  return Y;
}

template <class T>
std::ostream& operator<<( std::ostream& os, Matrix<T>& A ) {
  int n = A.getN();
  if( n > 1000 ) {
    return (os << "It cannot be printed\n");
  }
  T *M = (T *) malloc(n*n*sizeof(T));
  A.get(M);
  for( int i=0; i<n; i++ ) {
    char s[1200];
    int ind = 0;
    for( int j=0; j<n; j++ ) {
      ind += sprintf(&s[ind],"%12.4e",M[i+j*n]); 
    }
    sprintf(&s[ind],"\n");
    os << s;
  }
  free(M);
  return os;
}
#endif

