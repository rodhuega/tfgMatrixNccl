#pragma once

#include <iostream>
#include <vector>

#include "nccl.h"

#include "ErrorCheckingCuda.cuh"
/**
 * @brief Clase que contiene todos los elementos fundamentales de colores,rangos y comunicadores tanto lógicos como físicos
 * para realizar las comunicaciones en el algoritmo Summa
 * 
 */
class CommSummaElement
{
    private:
        int idGpuLogic,idGpuPhysical,rankCommRowPhysical,rankCommColumnPhysical,rankCommRowLogic,rankCommColumnLogic,rowColor,columnColor;
        ncclComm_t commRow,commColumn;
        cudaStream_t *streamRow,*streamColumn;
        std::vector<int> rowDevices,columnDevices;
    public:
        /**
         * @brief Constructor de CommSummaElement que contiene la información de colores,rangos y comunicadores
         * tanto lógicos como físicos para realizar las comunicaciones en el algoritmo Summa
         * 
         * @param idGpuLogic , id de la gpu lógica
         * @param idGpuPhysical , id de la gpu física
         * @param rowColor , color de la fila del elemento
         * @param columnColor , color de la columna del elemento
         */
        CommSummaElement(int idGpuLogic,int idGpuPhysical,int rowColor,int columnColor);
        /**
         * @brief Destructor del elemento CommSummaElement
         * 
         */
        ~CommSummaElement();
        /**
         * @brief Devuelve la id lógica del elemento
         * 
         * @return int 
         */
        int getIdLogic();
        /**
         * @brief Devuelve la id física del elemento
         * 
         * @return int 
         */
        int getIdPhysical();
        /**
         * @brief Devuelve el rango físico de la fila del elemento
         * 
         * @return int 
         */
        int getRankCommRowPhysical();
        /**
         * @brief Devuelve el rango físico de la columna del elemento
         * 
         * @return int 
         */
        int getRankCommColumnPhysical();
        /**
         * @brief Devuelve el rango lógico de la fila del elemento
         * 
         * @return int 
         */
        int getRankCommRowLogic();
        /**
         * @brief Devuelve el rango lógico de la columna del elemento
         * 
         * @return int 
         */
        int getRankCommColumnLogic();
        /**
         * @brief Devuelve el color de la fila del elemento
         * 
         * @return int 
         */
        int getRowColor();
        /**
         * @brief Devuelve el color de la columna del elemento
         * 
         * @return int 
         */
        int getColumnColor();
        /**
         * @brief Devuelve un vector con las id de todas las gpus lógicas con las que se tiene que comunicar en la fila
         * 
         * @return std::vector<int>  
         */
        std::vector<int> getRowDevices();
        /**
         * @brief Devuelve un vector con las id de todas las gpus lógicas con las que se tiene que comunicar en la columna
         * 
         * @return std::vector<int>  
         */
        std::vector<int> getColumnDevices();
        /**
         * @brief Devuelve el comunicador nccl de la fila
         * 
         * @return ncclComm_t 
         */
        ncclComm_t getCommRow();
        /**
         * @brief Devuelve el comunicador nccl de la columna
         * 
         * @return ncclComm_t 
         */
        ncclComm_t getCommColumn();
        /**
         * @brief Devuelve el puntero la stream para la fila
         * 
         * @return cudaStream_t* 
         */
        cudaStream_t* getStreamRow();
        /**
         * @brief Devuelve el puntero la stream para la columna
         * 
         * @return cudaStream_t* 
         */
        cudaStream_t* getStreamColumn();
        /**
         * @brief Asigna el rango físico de la fila del elemento
         * 
         * @param rankCommRowPhysical 
         */
        void setRankCommRowPhysical(int rankCommRowPhysical);
        /**
         * @brief Asigna el rango físico de la columna del elemento
         * 
         * @param rankCommColumnPhysical 
         */
        void setRankCommColumnPhysical(int rankCommColumnPhysical);
        /**
         * @brief Asigna el rango lógico de la fila del elemento
         * 
         * @param rankCommRowLogic 
         */
        void setRankCommRowLogic(int rankCommRowLogic);
        /**
         * @brief Asigna el rango lógico de la columna del elemento
         * 
         * @param rankCommColumnLogic 
         */
        void setRankCommColumnLogic(int rankCommColumnLogic);
        /**
         * @brief Asigna el vector de ids de gpus lógicas con las que se comunica el elemento en la fila
         * 
         * @param rowDevices 
         */
        void setRowDevices(std::vector<int> rowDevices);
        /**
         * @brief Asigna el vector de ids de gpus lógicas con las que se comunica el elemento en la columna
         * 
         * @param columnDevices 
         */
        void setColumnDevices( std::vector<int> columnDevices);
        /**
         * @brief Asgina el comunicador de la fila
         * 
         * @param commRow 
         */
        void setCommRow(ncclComm_t commRow);
        /**
         * @brief Asgina el comunicador de la columna
         * 
         * @param commColumn 
         */
        void setCommColumn(ncclComm_t commColumn);
        /**
         * @brief Asgina el puntero de la stream de la fila
         * 
         * @param streamRow 
         */
        void setStreamRow(cudaStream_t* streamRow);
        /**
         * @brief Asgina el puntero de la stream de la columna
         * 
         * @param streamColumn 
         */
        void setStreamColumn(cudaStream_t* streamColumn);
        
};