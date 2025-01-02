#ifndef STARFM_H
#define STARFM_H

#include <string>
#include "gdal.h"
#include <cmath>
#include "device_launch_parameters.h"

// 同时声明为 __host__ 和 __device__ 的 is_valid 函数
__host__ __device__ inline bool is_valid(float value) {
    return isfinite(value) && !isnan(value);
}

// CPU 版本函数声明
void blending(float* L1, float* M1, float* M2, float* L2, int width, int height, int w, float A, float error_lm, float error_mm, int class_num);
void runSTARFM_CPU(const std::string& landsatPath, const std::string& modisPath1, const std::string& modisPath2,
    const std::string& outputPath, int win_size, float L_err, float M_err, int class_num, float A);

// GPU 版本函数声明
void runSTARFM_GPU(const std::string& landsatPath, const std::string& modisPath1, const std::string& modisPath2, const std::string& outputPath,
    int win_size, float L_err, float M_err, int class_num, float A);

// 读取参数文件函数声明
bool readParamsFromFile(const std::string& filePath, int& win_size, float& L_err, float& M_err, int& class_num,
    float& A, std::string& landsatPath1, std::string& modisPath1,
    std::string& modisPath2, std::string& landsatPath2);

#endif // STARFM_H