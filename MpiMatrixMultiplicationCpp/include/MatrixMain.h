#ifndef MatrixMain_H
#define MatrixMain_H

#include <iostream>
#include <mpi.h>
#include "MatrixUtilities.h"

/**
 * @brief Clase que contiene la matriz completa y sus principales caracteristicas
 * 
 * @tparam Toperation , tipo de la matriz(double,int,float)
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
  Toperation *matrix;
  

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
   * @brief Obtiene el puntero de la matriz.
   * 
   * @return Toperation* 
   */
  Toperation *getMatrix();
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
   * @brief Rellena el atributo Toperation matrix de valores
   * 
   * @param filltype , Indica como se va a rellenar la matriz
   * @param matrixFromMemory , Puntero de la matriz que se va a asignar desde memoria en caso de que asi lo indique el fillType
   */
  void setMatrix(Toperation* newMatrix);
  /**
   * @brief Asigna si una matriz esta distribuida o no
   * 
   * @param isDistributed 
   */
  void setIsDistributed(bool isDistributed);

};
#endif