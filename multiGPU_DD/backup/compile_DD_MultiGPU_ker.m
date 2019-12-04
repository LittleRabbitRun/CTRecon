
system( '/usr/local/cuda/bin/nvcc --std=c++11 -Xcompiler -fopenmp -O3 --use_fast_math --compile   -o DD_MultiGPU_ker.o  --compiler-options -fPIC  -I"/usr/local/MATLAB/R2015b/extern/include " -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc "DD_MultiGPU_ker.cu" ' );
mex -v -largeArrayDims  COMPFLAGS="$COMPFLAGS -fopenmp" -L/usr/local/cuda/lib64 -lcudart -lgomp DD_MultiGPU_mex.cpp DD_MultiGPU_ker.o
