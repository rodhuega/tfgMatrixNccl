#include "CommSummaElement.cuh"


CommSummaElement::CommSummaElement(int idGpuLogic,int idGpuPhysical,int rowColor,int columnColor)
{
    this->idGpuLogic=idGpuLogic;
    this->idGpuPhysical=idGpuPhysical;
    this->rowColor=rowColor;
    this->columnColor=columnColor;
    this->lastRowMySelf=0;
    this->lastColumnMySelf=0;
    CUDACHECK(cudaSetDevice(idGpuPhysical));
    streamRow= new cudaStream_t;streamColumn= new cudaStream_t;
    CUDACHECK(cudaStreamCreate(streamRow));
    CUDACHECK(cudaStreamCreate(streamColumn));
}

CommSummaElement::~CommSummaElement()
{
    if(idGpuLogic==idGpuPhysical)
    {
        NCCLCHECK(ncclCommDestroy(commRow));
        NCCLCHECK(ncclCommDestroy(commColumn));
        if(streamRow!=nullptr)
        {
            CUDACHECK(cudaStreamDestroy(*streamRow));
        }
        if(streamColumn!=nullptr)
        {
            CUDACHECK(cudaStreamDestroy(*streamColumn));
        }
    }
}

int CommSummaElement::getIdLogic()
{
    return idGpuLogic;
}


int CommSummaElement::getIdPhysical()
{
    return idGpuPhysical;
}


int CommSummaElement::getRankCommRowPhysical()
{
    return rankCommRowPhysical;
}

int  CommSummaElement::getRankCommColumnPhysical()
{
    return rankCommColumnPhysical;
}

int CommSummaElement::getRankCommRowLogic()
{
    return rankCommRowLogic;
}

int CommSummaElement::getRankCommColumnLogic()
{
    return rankCommColumnLogic;
}

int CommSummaElement::getRowColor()
{
    return rowColor;
}

int CommSummaElement::getColumnColor()
{
    return columnColor;
}

std::vector<std::vector<int>> CommSummaElement::getRowDevices()
{
    return rowDevices;
}
std::vector<std::vector<int>> CommSummaElement::getColumnDevices()
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

ncclComm_t CommSummaElement::getCommRowMySelf()
{
    return commsRowsMySelf[lastRowMySelf++];
}

ncclComm_t CommSummaElement::getCommColumnMySelf()
{
    return commsColumnsMySelf[lastColumnMySelf++];
}

cudaStream_t* CommSummaElement::getStreamRow()
{
    return streamRow;
}

cudaStream_t* CommSummaElement::getStreamColumn()
{
    return streamColumn;
}

cudaStream_t* CommSummaElement::getStreamRowMySelf()
{
    return streamsRowsMySelf[lastRowMySelf];
}

cudaStream_t* CommSummaElement::getStreamColumnMySelf()
{
    return streamsColumnsMySelf[lastColumnMySelf];
}

void CommSummaElement::setRankCommRowPhysical(int rankCommRowPhysical)
{
    this->rankCommRowPhysical=rankCommRowPhysical;
}

void CommSummaElement::setRankCommColumnPhysical(int rankCommColumnPhysical)
{
    this->rankCommColumnPhysical=rankCommColumnPhysical;
}

void CommSummaElement::setRankCommRowLogic(int rankCommRowLogic)
{
    this->rankCommRowLogic=rankCommRowLogic;
}

void CommSummaElement::setRankCommColumnLogic(int rankCommColumnLogic)
{
    this->rankCommColumnLogic=rankCommColumnLogic;
}

void CommSummaElement::setRowDevices(std::vector<std::vector<int>> rowDevices)
{
    this->rowDevices=rowDevices;
}
        
void CommSummaElement::setColumnDevices(std::vector<std::vector<int>> columnDevices)
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

void CommSummaElement::addCommRowMySelf(ncclComm_t commRowMySelf)
{
    this->commsRowsMySelf.push_back(commRowMySelf);
    cudaStream_t * newStream=new cudaStream_t;
    CUDACHECK(cudaStreamCreate(newStream));
    streamsRowsMySelf.push_back(newStream);
}

void CommSummaElement::addCommColumnMySelf(ncclComm_t commColumnMySelf)
{
    this->commsColumnsMySelf.push_back(commColumnMySelf);
    cudaStream_t * newStream=new cudaStream_t;
    CUDACHECK(cudaStreamCreate(newStream));
    streamsColumnsMySelf.push_back(newStream);
}

void CommSummaElement::setStreamRow(cudaStream_t* streamRow)
{
    this->streamRow=streamRow;
}

void CommSummaElement::setStreamColumn(cudaStream_t* streamColumn)
{
    this->streamColumn=streamColumn;
}

void CommSummaElement::waitStreams()
{
    int i;
    for(i=0;i<lastColumnMySelf;i++)
    {
        CUDACHECK(cudaStreamSynchronize(*streamsColumnsMySelf[i]));
    }
    for(i=0;i<lastRowMySelf;i++)
    {
        CUDACHECK(cudaStreamSynchronize(*streamsRowsMySelf[i]));
    }
    lastColumnMySelf=0;
    lastRowMySelf=0;
    CUDACHECK(cudaStreamSynchronize(*streamRow));
    CUDACHECK(cudaStreamSynchronize(*streamColumn));
    
}