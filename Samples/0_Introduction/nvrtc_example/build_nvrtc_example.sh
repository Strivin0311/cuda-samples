SOURCE=nvrtc_example.cpp
BUILD_PATH=./build/nvrtc_example

nvcc -rdc=true -ccbin g++ -gencode arch=compute_90,code=sm_90 \
    -lnccl -lcudart -lcuda -lnvrtc \
    -o $BUILD_PATH $SOURCE