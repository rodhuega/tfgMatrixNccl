#include "CommSummaElement.cuh"


CommSummaElement::CommSummaElement(int idGpuLogic,int idGpuPhysical,int rowColor,int columnColor)
{
    this->idGpuLogic=idGpuLogic;
    this->idGpuPhysical=idGpuPhysical;
    this->rowColor=rowColor;
    this->columnColor=columnColor;
}

int CommSummaElement::getIdLogic()
{
    return idGpuLogic;
}


int CommSummaElement::getIdPhysical()
{
    return idGpuPhysical;
}


int CommSummaElement::getRankCommRow()
{
    return rankCommRow;
}

int CommSummaElement::getRankCommColumn()
{
    return rankCommColumn;
}

int CommSummaElement::getRowColor()
{
    return rowColor;
}

int CommSummaElement::getColumnColor()
{
    return columnColor;
}

std::vector<int> CommSummaElement::getRowDevices()
{
    return rowDevices;
}
std::vector<int> CommSummaElement::getColumnDevices()
{
    return columnDevices;
}

ncclComm_t CommSummaElement::getCommRow()
{
    return commRow;
}

ncclComm_t CommSummaElement::getCommColumn()
{
    return commColumn;
}

void CommSummaElement::setRankCommRow(int rankCommRow)
{
    this->rankCommRow=rankCommRow;
}

void CommSummaElement::setRankCommColumn(int rankCommColumn)
{
    this->rankCommColumn=rankCommColumn;
}

void CommSummaElement::setRowDevices(std::vector<int> rowDevices)
{
    this->rowDevices=rowDevices;
}
        
void CommSummaElement::setColumnDevices( std::vector<int> columnDevices)
{
    this->columnDevices=columnDevices;
}

void CommSummaElement::setCommRow(ncclComm_t commRow)
{
    this->commRow=commRow;
}

void CommSummaElement::setCommColumn(ncclComm_t commColumn)
{
    this->commColumn=commColumn;
}