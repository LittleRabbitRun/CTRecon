system( '/usr/local/cuda/bin/nvcc -std=c++11 -Xcompiler -fopenmp -O3 --use_fast_math --compile -o KatsevichBackprojectionGPU.o  --compiler-options -fPIC -I"/usr/local/cuda/include"  "KatsevichBackprojectionGPU.cu" ');
mex -v -largeArrayDims  GCC='/usr/bin/gcc' COMPFLAGS="$COMPFLAGS -fopenmp -std=c++11" -L/usr/local/cuda/lib64 -L"/usr/local/cuda/lib64" -lcudart KatsevichBkProj.cpp KatsevichBackprojectionGPU.o 

c = FunctionKat(0.5, 0, 0);