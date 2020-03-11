import sys
import os
import subprocess


def main():
    minCpu=int(sys.argv[1])
    maxCpu=int(sys.argv[2])
    stepCpu=int(sys.argv[3])
    minM=int(sys.argv[4])
    maxM=int(sys.argv[5])
    stepM=int(sys.argv[6])
    minN=int(sys.argv[7])
    maxN=int(sys.argv[8])
    stepN=int(sys.argv[9])
    minK=int(sys.argv[10])
    maxK=int(sys.argv[11])
    stepK=int(sys.argv[12])
    lowerRandom=int(sys.argv[13])
    upperRandom=int(sys.argv[14])
    makeOutput = subprocess.check_output(f"make", shell=True)
    print(makeOutput.decode("ascii",errors="ignore"))
    path = os.path.abspath("./bin/main")
    for m in range(minM,maxM+1,stepM):
        for n in range(minN,maxN+1,stepN):
            for k in range(minK,maxK+1,stepK):
                for cpu in range(minCpu,maxCpu+1,stepCpu):
                    print(f"Numero de cpus usadas: {cpu}, M: {m}, N: {n}, K: {k}")
                    output = subprocess.check_output(f"mpirun --oversubscribe -np {cpu} {path} -r {m} {n} {k} {lowerRandom} {upperRandom}", shell=True)
                    outputString=output.decode("ascii",errors="ignore")
                    for line in outputString.split("\n"):
                        print(line)
                    print("----------------------------------------------------------------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()