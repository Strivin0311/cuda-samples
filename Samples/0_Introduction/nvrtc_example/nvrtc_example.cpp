#include <iostream>
#include <nvrtc.h>
#include <cuda_runtime.h>
#include <cuda.h>

// 定义初始的 CUDA 内核代码
const char* initial_kernel_code = R"(
extern "C" __global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
)";

// 定义修改后的 CUDA 内核代码
const char* modified_kernel_code = R"(
extern "C" __global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
    printf("Thread %d: a[%d] + b[%d] = c[%d] = %d\n", i, i, i, i, c[i]);
}
)";

void compileAndRunKernel(const char* kernel_code) {
    // 初始化 NVRTC
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, kernel_code, "addKernel", 0, NULL, NULL);

    // 编译 CUDA 内核代码为 PTX
    size_t ptxSize;
    nvrtcCompileProgram(prog, 0, NULL);
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // 初始化 CUDA 驱动 API
    cuInit(0);
    CUcontext context;
    CUdevice device;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // 加载 PTX 模块
    CUmodule module;
    cuModuleLoadData(&module, ptx);

    // 获取内核函数的句柄
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "addKernel");

    // 分配设备内存
    int a[256], b[256], c[256];
    CUdeviceptr d_a, d_b, d_c;
    cuMemAlloc(&d_a, 256 * sizeof(int));
    cuMemAlloc(&d_b, 256 * sizeof(int));
    cuMemAlloc(&d_c, 256 * sizeof(int));

    // 初始化输入数据
    for (int i = 0; i < 256; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // 将输入数据拷贝到设备
    cuMemcpyHtoD(d_a, a, 256 * sizeof(int));
    cuMemcpyHtoD(d_b, b, 256 * sizeof(int));

    // 启动 CUDA 内核
    void* args[] = { &d_c, &d_a, &d_b };
    cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, 0, args, 0);

    // 将结果拷贝回主机
    cuMemcpyDtoH(c, d_c, 256 * sizeof(int));

    // 输出结果
    for (int i = 0; i < 256; ++i) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }

    // 释放资源
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    delete[] ptx;
    nvrtcDestroyProgram(&prog);
}

int main() {
    // 运行初始版本的内核
    std::cout << "Running initial kernel..." << std::endl;
    compileAndRunKernel(initial_kernel_code);

    // 运行修改版本的内核
    std::cout << "\nRunning modified kernel..." << std::endl;
    compileAndRunKernel(modified_kernel_code);

    return 0;
}