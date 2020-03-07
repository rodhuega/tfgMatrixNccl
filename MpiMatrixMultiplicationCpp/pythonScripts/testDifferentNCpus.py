import sys
import os


def main():
    minCpu=int(sys.argv[1])
    maxCpu=int(sys.argv[2])
    stepCpu=int(sys.argv[3])
    M=int(sys.argv[4])
    N=int(sys.argv[5])
    K=int(sys.argv[6])
    lowerRandom=int(sys.argv[7])
    upperRandom=int(sys.argv[8])
    path = os.path.abspath("./bin/main")
    for cpu in range(minCpu,maxCpu+1,stepCpu):
        print(f"Numero de cpus usadas: {cpu}, M: {M}, N: {N}, K: {K}")
        os.system(f"mpirun -np {cpu} {path} -r {M} {N} {K} {lowerRandom} {upperRandom}")
        print("----------------------------------------------------------------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()