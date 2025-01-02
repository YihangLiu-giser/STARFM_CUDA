#include "starfm.h"
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <params_file> <cpu/gpu>" << std::endl;
        return 1;
    }

    // 读取参数文件路径
    std::string paramsFilePath = argv[1];
    std::string mode = argv[2]; // "cpu" or "gpu"

    // 从文件中读取参数
    int win_size;
    float L_err, M_err, A;
    int class_num;
    std::string landsatPath1, modisPath1, modisPath2, landsatPath2;

    if (!readParamsFromFile(paramsFilePath, win_size, L_err, M_err, class_num, A,
        landsatPath1, modisPath1, modisPath2, landsatPath2)) {
        std::cerr << "Error reading parameters from file." << std::endl;
        return 1;
    }

    // 根据用户选择执行CPU或GPU版本
    if (mode == "cpu") {
        runSTARFM_CPU(landsatPath1, modisPath1, modisPath2, landsatPath2, win_size, L_err, M_err, class_num, A);
        std::cout << "SATRFM算法执行完毕（CPU），图像保存为: " << landsatPath2  << std::endl;
    }
    else if (mode == "gpu") {
        runSTARFM_GPU(landsatPath1, modisPath1, modisPath2, landsatPath2, win_size, L_err, M_err, class_num, A);
        std::cout << "SATRFM算法执行完毕（GPU），图像保存为: " << landsatPath2  << std::endl;
    }
    else {
        std::cerr << "Invalid mode. Choose 'cpu' or 'gpu'." << std::endl;
        return 1;
    }

    return 0;
}