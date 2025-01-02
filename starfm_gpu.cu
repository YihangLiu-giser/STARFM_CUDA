#include "starfm.h"
#include "gdal_priv.h"
#include "gdalwarper.h"
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"

// 定义max和min函数 (GPU版本)
__device__ int device_max(int a, int b) {
    return (a > b) ? a : b;
}

__device__ int device_min(int a, int b) {
    return (a < b) ? a : b;
}

// CUDA 内核函数
__global__ void blending_kernel(float* L1, float* M1, float* M2, float* L2, int width, int height, int w, float A, float error_lm, float error_mm, int class_num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        float r_LM, r_MM, r_center_LM, r_center_MM;
        float sum1 = 0, sum2 = 0;
        int rmin, rmax, smin, smax;
        float result = 0;
        int judge = 0;
        int kk = 0;
        float wei = 0;

        // 检查每个像素值是否有效
        if (!is_valid(M2[idx]) || !is_valid(L1[idx]) || !is_valid(M1[idx])) {
            return;  // 跳过无效值
        }

        r_center_LM = M2[idx] - L1[idx] + error_lm;
        r_center_MM = M2[idx] - M1[idx] + error_mm;

        // 定义窗口大小
        rmin = device_max(col - w / 2, 0);
        rmax = device_min(col + w / 2, width - 1);
        smin = device_max(row - w / 2, 0);
        smax = device_min(row + w / 2, height - 1);

        // 局部窗口内计算Landsat1
        for (int i = smin; i <= smax; i++) {
            for (int j = rmin; j <= rmax; j++) {
                if (is_valid(L1[i * width + j])) {
                    sum1 += L1[i * width + j] * L1[i * width + j];
                    sum2 += L1[i * width + j];
                }
            }
        }

        // 计算标准差，确保sum1和sum2有效
        if (sum1 == 0 || sum2 == 0) {
            return;  // 避免计算错误，跳过此像素
        }

        float sigma = sqrt(sum1 / (w * w) - (sum2 / (w * w)) * (sum2 / (w * w)));  // 标准差
        float st = sigma / class_num;

        // 循环计算邻域像素的加权结果
        for (int i = smin; i <= smax; i++) {
            for (int j = rmin; j <= rmax; j++) {
                if (is_valid(L1[i * width + j]) && fabs(L1[idx] - L1[i * width + j]) < st) {
                    r_LM = M2[i * width + j] - L1[i * width + j];
                    r_MM = M2[i * width + j] - M1[i * width + j];
                    if ((r_center_LM > 0 && r_LM < r_center_LM) || (r_center_LM < 0 && r_LM > r_center_LM)) {
                        if ((r_center_MM > 0 && r_MM < r_center_MM) || (r_center_MM < 0 && r_MM > r_center_MM)) {
                            r_LM = fabs(r_LM) + 0.0001;
                            r_MM = fabs(r_MM) + 0.0001;
                            if (idx == i * width + j) {
                                judge = 1;
                            }
                            float dis = sqrtf((row - i) * (row - i) + (col - j) * (col - j));
                            dis = dis / A + 1.0;
                            float weih = 1.0 / (dis * r_LM * r_MM);
                            wei += weih;
                            result += weih * (M1[i * width + j] + L1[i * width + j] - M2[i * width + j]);
                            kk++;
                        }
                    }
                }
            }
        }

        if (kk == 0) {
            L2[idx] = fabs(L1[idx] + M1[idx] - M2[idx]);
            wei = 10000;
        }
        else {
            if (judge == 0) {
                float dis = 1.0;
                r_LM = fabs(M2[idx] - L1[idx]) + 0.0001;
                r_MM = fabs(M2[idx] - M1[idx]) + 0.0001;
                float weih = 1.0 / (dis * r_LM * r_MM);
                result += weih * (L1[idx] + M1[idx] - M2[idx]);
                wei += weih;
            }
            L2[idx] = result / wei;
        }
    }
}

// 用于获取当前CUDA内存状态
void printCudaMemUsage() {
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "CUDA Memory Usage: " << std::endl;
    std::cout << "Free Memory: " << freeMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total Memory: " << totalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Used Memory: " << (totalMem - freeMem) / (1024 * 1024) << " MB" << std::endl;
}

// 执行算法函数
void runSTARFM_GPU(const std::string& landsatPath, const std::string& modisPath1, const std::string& modisPath2, const std::string& outputPath,
    int win_size, float L_err, float M_err, int class_num, float A) {
    GDALAllRegister();

    // 开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 打开Landsat和MODIS数据集
    GDALDataset* pLandsatDataset = (GDALDataset*)GDALOpen(landsatPath.c_str(), GA_ReadOnly);
    GDALDataset* pModisDataset1 = (GDALDataset*)GDALOpen(modisPath1.c_str(), GA_ReadOnly);
    GDALDataset* pModisDataset2 = (GDALDataset*)GDALOpen(modisPath2.c_str(), GA_ReadOnly);

    if (!pLandsatDataset || !pModisDataset1 || !pModisDataset2) {
        std::cerr << "Error opening datasets." << std::endl;
        return;
    }

    int width = pLandsatDataset->GetRasterXSize();
    int height = pLandsatDataset->GetRasterYSize();
    int landsatBands = pLandsatDataset->GetRasterCount();  // Landsat波段数
    int modisBands = pModisDataset1->GetRasterCount();  // MODIS波段数

    // 在GPU上分配内存
    float* d_Landsat1;
    float* d_Modis1;
    float* d_Modis2;
    float* d_Landsat2;

    cudaMalloc(&d_Landsat1, width * height * sizeof(float));
    cudaMalloc(&d_Modis1, width * height * sizeof(float));
    cudaMalloc(&d_Modis2, width * height * sizeof(float));
    cudaMalloc(&d_Landsat2, width * height * sizeof(float));

    // 打印CUDA内存使用情况
    printCudaMemUsage();

    // 创建输出数据集
    GDALDriver* pDriver = pLandsatDataset->GetDriver();
    GDALDataset* pOutputDataset = pDriver->Create(outputPath.c_str(), width, height, landsatBands, GDT_Float32, nullptr);

    if (!pOutputDataset) {
        std::cerr << "Error creating output dataset." << std::endl;
        return;
    }

    // 处理每个波段
    for (int band = 1; band <= landsatBands; band++) {
        // 读取Landsat波段数据
        GDALRasterBand* pLandsatBand = pLandsatDataset->GetRasterBand(band);
        std::vector<float> BufferLandsat1(width * height);
        pLandsatBand->RasterIO(GF_Read, 0, 0, width, height, &BufferLandsat1[0], width, height, GDT_Float32, 0, 0);

        // 读取MODIS波段数据（假设MODIS具有相同的波段数量）
        GDALRasterBand* pModisBand1 = pModisDataset1->GetRasterBand(band);
        std::vector<float> BufferModis1(width * height);
        pModisBand1->RasterIO(GF_Read, 0, 0, width, height, &BufferModis1[0], width, height, GDT_Float32, 0, 0);

        GDALRasterBand* pModisBand2 = pModisDataset2->GetRasterBand(band);
        std::vector<float> BufferModis2(width * height);
        pModisBand2->RasterIO(GF_Read, 0, 0, width, height, &BufferModis2[0], width, height, GDT_Float32, 0, 0);

        // 将数据从主机内存复制到设备内存
        cudaMemcpy(d_Landsat1, &BufferLandsat1[0], width * height * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Modis1, &BufferModis1[0], width * height * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Modis2, &BufferModis2[0], width * height * sizeof(float), cudaMemcpyHostToDevice);

        // 运行CUDA内核
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        blending_kernel << <grid, block >> > (d_Landsat1, d_Modis1, d_Modis2, d_Landsat2, width, height, win_size, A, L_err, M_err, class_num);
        cudaDeviceSynchronize();

        // 将数据从设备内存复制回主机内存
        std::vector<float> BufferLandsat2(width * height);
        cudaMemcpy(&BufferLandsat2[0], d_Landsat2, width * height * sizeof(float), cudaMemcpyDeviceToHost);

        // 写入输出数据
        GDALRasterBand* pOutBand = pOutputDataset->GetRasterBand(band);
        pOutBand->RasterIO(GF_Write, 0, 0, width, height, &BufferLandsat2[0], width, height, GDT_Float32, 0, 0);
    }

    // 打印CUDA内存使用情况
    printCudaMemUsage();

    // 清理和释放资源
    cudaFree(d_Landsat1);
    cudaFree(d_Modis1);
    cudaFree(d_Modis2);
    cudaFree(d_Landsat2);

    GDALClose(pLandsatDataset);
    GDALClose(pModisDataset1);
    GDALClose(pModisDataset2);
    GDALClose(pOutputDataset);

    // 结束时间
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Processing time: " << duration.count() << " seconds" << std::endl;
}