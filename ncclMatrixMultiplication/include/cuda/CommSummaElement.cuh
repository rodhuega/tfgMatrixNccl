#pragma once

#include <vector>

#include "nccl.h"


class CommSummaElement
{
    private:
        int idGpuLogic,idGpuPhysical,rankCommRow,rankCommColumn,rowColor,columnColor;
        ncclComm_t commRow,commColumn;
        std::vector<int> rowDevices,columnDevices;
    public:
        CommSummaElement(int idGpuLogic,int idGpuPhysical,int rowColor,int columnColor);

        int getIdLogic();
        int getIdPhysical();
        int getRankCommRow();
        int getRankCommColumn();
        int getRowColor();
        int getColumnColor();
        std::vector<int> getRowDevices();
        std::vector<int> getColumnDevices();
        ncclComm_t getCommRow();
        ncclComm_t getCommColumn();
        void setRankCommRow(int rankCommRow);

        void setRankCommColumn(int rankCommColumn);

        void setRowDevices(std::vector<int> rowDevices);

        void setColumnDevices( std::vector<int> columnDevices);

        void setCommRow(ncclComm_t commRow);

        void setCommColumn(ncclComm_t commColumn);
        

        
};