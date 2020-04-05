#pragma once

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "nccl.h"

#include <unordered_map>
#include <iostream>
#include <tuple>
#include <string>

#include "OperationProperties.h"
#include "OperationType.h"

#include "ErrorCheckingCuda.cuh"


template <class Toperation>
class NcclMultiplicationEnvironment
{
private:
    ncclComm_t commWorld,commOperation;
    ncclDataType_t basicOperationType;
    OperationType opType;
    int gpuSizeOperation,gpuSizeSystem,gpuSizeWorld,gpuRoot;
    // std::unordered_map<std::string,MatrixMain<Toperation>*> matricesMatrixMain;
    // std::unordered_map<std::string,Toperation*> matricesGlobalSimplePointer;
    // std::unordered_map<std::string,dimensions> matricesGlobalDimensions;

public:

    NcclMultiplicationEnvironment(int gpuSizeWorld,int gpuRoot,OperationType opType);


};