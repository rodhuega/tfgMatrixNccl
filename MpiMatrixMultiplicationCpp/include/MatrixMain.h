#ifndef MatrixMain_H
#define MatrixMain_H

#include <iostream>
#include <fstream>
#include <random>
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
  int boundLower;
  int boundUpper;
  Toperation *matrix;
  std::ifstream file;
  

public:

  /**
   * @brief Se construye la matriz de forma aleatoria a partir de los valores que se pasan como argumentos
   * 
   * @param rows , filas de la matriz
   * @param columns , columnas de la matriz
   * @param boundLower , limite inferior de los numeros aleatorios que estaran en la matriz
   * @param boundUpper ,  limite superior de los numeros aleatorios que estaran en la matriz
   */
  MatrixMain(int rows, int columns, int boundLower, int boundUpper);
  /**
   * @brief Se crea el objeto a partir de la lectura de un fichero de texto
   * 
   * @param filename 
   */
  MatrixMain(const char *filename);
  /**
   * @brief Destructor del objeto
   * 
   */
  ~MatrixMain();
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
   * @param isRandom , En caso de ser verdadero lo rellena de valores aleatorios, en caso contrario los lee de un fichero
   */
  void fillMatrix(bool isRandom);

};
#endif