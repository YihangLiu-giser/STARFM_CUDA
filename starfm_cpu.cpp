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
#include <fstream>
#include <sstream>

void blending(float* L1, float* M1, float* M2, float* L2, int width, int height, int w, float A, float error_lm, float error_mm, int class_num)
{
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
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
                continue;  // 跳过无效值
            }

            r_center_LM = M2[idx] - L1[idx] + error_lm;
            r_center_MM = M2[idx] - M1[idx] + error_mm;

            // 定义窗口大小
            rmin = std::max(col - w / 2, 0);
            rmax = std::min(col + w / 2, width - 1);
            smin = std::max(row - w / 2, 0);
            smax = std::min(row + w / 2, height - 1);

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
                continue;  // 避免计算错误，跳过此像素
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
                                float dis = sqrt((row - i) * (row - i) + (col - j) * (col - j));
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

            //// 调试：输出结果值的范围和一些像素值
            //if ((row * width + col) % (width * height / 10) == 0) {
            //    std::cout << "Debug: Pixel [" << row << ", " << col << "] result: " << L2[idx] << std::endl;
            //    std::cout << "    r_center_LM: " << r_center_LM << ", r_center_MM: " << r_center_MM << std::endl;
            //    std::cout << "    Sum1: " << sum1 << ", Sum2: " << sum2 << ", St: " << st << std::endl;
            //}
        }
    }
}

void runSTARFM_CPU(const std::string& landsatPath, const std::string& modisPath1, const std::string& modisPath2,
    const std::string& outputPath, int win_size, float L_err, float M_err, int class_num, float A) {

    GDALAllRegister();

    // 开始时间
    auto start = std::chrono::high_resolution_clock::now();

    GDALDataset* pLandsatDataset = (GDALDataset*)GDALOpen(landsatPath.c_str(), GA_ReadOnly);
    GDALDataset* pModisDataset1 = (GDALDataset*)GDALOpen(modisPath1.c_str(), GA_ReadOnly);
    GDALDataset* pModisDataset2 = (GDALDataset*)GDALOpen(modisPath2.c_str(), GA_ReadOnly);

    if (!pLandsatDataset || !pModisDataset1 || !pModisDataset2) {
        std::cerr << "Error opening datasets." << std::endl;
        return;
    }

    int width = pLandsatDataset->GetRasterXSize();
    int height = pLandsatDataset->GetRasterYSize();
    int landsatBands = pLandsatDataset->GetRasterCount();  // Landsat的波段数量
    int modisBands = pModisDataset1->GetRasterCount();  // MODIS的波段数量（假设两个MODIS数据集具有相同波段数量）

    // 开辟内存
    std::vector<float> BufferLandsat1(width * height);
    std::vector<float> BufferModis1(width * height);
    std::vector<float> BufferModis2(width * height);
    std::vector<float> BufferLandsat2(width * height);

    // 创建输出数据集
    GDALDriver* pDriver = pLandsatDataset->GetDriver();  // 使用与Landsat相同的驱动
    GDALDataset* pOutputDataset = pDriver->Create(outputPath.c_str(), width, height, landsatBands, GDT_Float32, nullptr);

    if (!pOutputDataset) {
        std::cerr << "Error creating output dataset." << std::endl;
        return;
    }

    // 处理每个波段
    for (int band = 1; band <= landsatBands; band++) {
        // 读取Landsat波段数据
        GDALRasterBand* pLandsatBand = pLandsatDataset->GetRasterBand(band);
        pLandsatBand->RasterIO(GF_Read, 0, 0, width, height, &BufferLandsat1[0], width, height, GDT_Float32, 0, 0);

        // 读取MODIS波段数据（假设MODIS具有相同的波段数量）
        GDALRasterBand* pModisBand1 = pModisDataset1->GetRasterBand(band);
        pModisBand1->RasterIO(GF_Read, 0, 0, width, height, &BufferModis1[0], width, height, GDT_Float32, 0, 0);

        GDALRasterBand* pModisBand2 = pModisDataset2->GetRasterBand(band);
        pModisBand2->RasterIO(GF_Read, 0, 0, width, height, &BufferModis2[0], width, height, GDT_Float32, 0, 0);

        // 执行 blending 函数
        blending(&BufferLandsat1[0], &BufferModis1[0], &BufferModis2[0], &BufferLandsat2[0],
            width, height, win_size, A, L_err, M_err, class_num);

        // 写入输出数据
        GDALRasterBand* pOutBand = pOutputDataset->GetRasterBand(band);
        pOutBand->RasterIO(GF_Write, 0, 0, width, height, &BufferLandsat2[0], width, height, GDT_Float32, 0, 0);
    }

    // 关闭数据集
    GDALClose(pLandsatDataset);
    GDALClose(pModisDataset1);
    GDALClose(pModisDataset2);
    GDALClose(pOutputDataset);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Processing time: " << elapsed.count() << " seconds." << std::endl;
}

// 读取参数文件
bool readParamsFromFile(const std::string& filePath, int& win_size, float& L_err, float& M_err, int& class_num,
    float& A, std::string& landsatPath1, std::string& modisPath1,
    std::string& modisPath2, std::string& landsatPath2) {
    std::ifstream file(filePath);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening parameter file." << std::endl;
        return false;
    }

    // 读取文件内容并解析
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string key, value;
        if (std::getline(ss, key, '=') && std::getline(ss, value)) {
            if (key == "win_size") win_size = std::stoi(value);
            else if (key == "L_err") L_err = std::stof(value);
            else if (key == "M_err") M_err = std::stof(value);
            else if (key == "class_num") class_num = std::stoi(value);
            else if (key == "A") A = std::stof(value);
            else if (key == "landsatPath1") landsatPath1 = value;
            else if (key == "modisPath1") modisPath1 = value;
            else if (key == "modisPath2") modisPath2 = value;
            else if (key == "landsatPath2") landsatPath2 = value;
        }
    }
    return true;
}