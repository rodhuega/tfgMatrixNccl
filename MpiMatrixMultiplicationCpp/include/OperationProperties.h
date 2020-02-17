#ifndef OperationProperties_H
#define OperationProperties_H

struct OperationProperties
{
    int meshRowSize;
    int meshColumnSize;
    int rowsA;
    int columnsAorRowsB;
    int columnsB;
    int numberOf0;
    int cpuSize;
    int blockColumnSizeA;
    int blockRowSizeB;
    bool candidate;
};



#endif