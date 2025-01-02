#ifndef STARFM_H
#define STARFM_H

#include <string>
#include "gdal.h"
#include <cmath>
#include "device_launch_parameters.h"

// ͬʱ����Ϊ __host__ �� __device__ �� is_valid ����
__host__ __device__ inline bool is_valid(float value) {
    return isfinite(value) && !isnan(value);
}

// CPU �汾��������
void blending(float* L1, float* M1, float* M2, float* L2, int width, int height, int w, float A, float error_lm, float error_mm, int class_num);
void runSTARFM_CPU(const std::string& landsatPath, const std::string& modisPath1, const std::string& modisPath2,
    const std::string& outputPath, int win_size, float L_err, float M_err, int class_num, float A);

// GPU �汾��������
void runSTARFM_GPU(const std::string& landsatPath, const std::string& modisPath1, const std::string& modisPath2, const std::string& outputPath,
    int win_size, float L_err, float M_err, int class_num, float A);

// ��ȡ�����ļ���������
bool readParamsFromFile(const std::string& filePath, int& win_size, float& L_err, float& M_err, int& class_num,
    float& A, std::string& landsatPath1, std::string& modisPath1,
    std::string& modisPath2, std::string& landsatPath2);

#endif // STARFM_H