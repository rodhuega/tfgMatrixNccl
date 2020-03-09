#ifndef MatrixMain_H
#define MatrixMain_H

#include <iostream>
#include <mpi.h>
#include "MatrixUtilities.h"

template <class Toperation>
class MpiMatrix;

/**
 * @brief Clase que contiene la matriz completa y sus principales caracteristicas
 * 
 * @tparam Toperation , tipo de la matriz(double,float)
 */
template <class Toperation>
class MatrixMain
{
private:
  int rowsReal;
  int rowsUsed;
  int columnsReal;
  int columnsUsed;
  bool isDistributed;
  bool isMatrixGlobalHere;
  Toperation *matrixGlobal;
  MpiMatrix<Toperation> *matrixLocal;
  

public:

  /**
   * @brief Se construye la matriz de forma aleatoria a partir de los valores que se pasan como argumentos
   * 
   * @param rows , filas de la matriz
   * @param columns , columnas de la matriz
   */
  MatrixMain(int rows, int columns);
  /**
   * @brief Destructor del objeto
   * 
   */
  ~MatrixMain();
  /**
   * @brief Indica si una matriz esta distribuida o no.
   * 
   * @return true 
   * @return false 
   */
  bool getIsDistributed();
  /**
   * @brief Obtiene el valor de filas reales de la verdadera matriz
   * 
   * @return int 
   */
  int getRowsReal();
  /**
   * @brief Obtiene el valor de filas que se usara para operar, en caso de con coincidir con columnsReal significa que el exceso son 0
   * 
   * @return int 
   */
  int getRowsUsed();
  /**
   * @brief Obtiene el valor de columnas reales de la verdadera matriz
   * 
   * @return int 
   */
  int getColumnsReal();
  /**
   * @brief Obtiene el valor de columnas que se usara para operar, en caso de con coincidir con columnsReal significa que el exceso son 0
   * 
   * @return int 
   */
  int getColumnsUsed();
  /**
   * @brief Indica si hay una matriz global
   * 
   * @return int 
   */
  bool getIsMatrixGlobalHere();
  /**
   * @brief Obtiene el puntero de la matriz global.
   * 
   * @return Toperation* 
   */
  Toperation *getMatrix();
  /**
   * @brief Obtiene el puntero de la matriz Mpi local.
   * 
   * @return Toperation* 
   */
  MpiMatrix<Toperation>* getMpiMatrix();
  /**
   * @brief Asigna el valor de filas que se usara para operar, en caso de con coincidir con columnsReal significa que el exceso son 0
   * 
   * @param rowsUsed 
   */
  void setRowsUsed(int rowsUsed);
  /**
   * @brief Asigna el valor de columnas que se usara para operar, en caso de con coincidir con columnsReal significa que el exceso son 0
   * 
   * @param columnsUsed 
   */
  void setColumnsUsed(int columnsUsed);
  /**
   * @brief Asigna si hay una matriz global
   * 
   * @param columnsUsed 
   */
  void setIsMatrixGlobalHere(bool isMatrixGlobalHere);
  /**
   * @brief Rellena el atributo Toperation matrixGlobal de valores
   * 
   * @param matrixGlobalNew , Puntero de la matriz global que se va a asignar
   */
  void setMatrix(Toperation* matrixGlobalNew);
  /**
   * @brief Rellena el atributo MpiMatrix<Toperation>* matrixLocal de una matrizLocal
   * 
   * @param MpiMatrix<Toperation>* , Puntero de la matriz global que se va a asignar
   */
  void setMpiMatrix(MpiMatrix<Toperation>* matrixLocal);
  /**
   * @brief Asigna si una matriz esta distribuida o no
   * 
   * @param isDistributed 
   */
  void setIsDistributed(bool isDistributed);
  /**
   * @brief Elimina la matriz global de memoria
   * 
   */
  void eraseMatrixGlobal();


};
#endif