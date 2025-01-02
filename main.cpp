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

    // ��ȡ�����ļ�·��
    std::string paramsFilePath = argv[1];
    std::string mode = argv[2]; // "cpu" or "gpu"

    // ���ļ��ж�ȡ����
    int win_size;
    float L_err, M_err, A;
    int class_num;
    std::string landsatPath1, modisPath1, modisPath2, landsatPath2;

    if (!readParamsFromFile(paramsFilePath, win_size, L_err, M_err, class_num, A,
        landsatPath1, modisPath1, modisPath2, landsatPath2)) {
        std::cerr << "Error reading parameters from file." << std::endl;
        return 1;
    }

    // �����û�ѡ��ִ��CPU��GPU�汾
    if (mode == "cpu") {
        runSTARFM_CPU(landsatPath1, modisPath1, modisPath2, landsatPath2, win_size, L_err, M_err, class_num, A);
        std::cout << "SATRFM�㷨ִ����ϣ�CPU����ͼ�񱣴�Ϊ: " << landsatPath2  << std::endl;
    }
    else if (mode == "gpu") {
        runSTARFM_GPU(landsatPath1, modisPath1, modisPath2, landsatPath2, win_size, L_err, M_err, class_num, A);
        std::cout << "SATRFM�㷨ִ����ϣ�GPU����ͼ�񱣴�Ϊ: " << landsatPath2  << std::endl;
    }
    else {
        std::cerr << "Invalid mode. Choose 'cpu' or 'gpu'." << std::endl;
        return 1;
    }

    return 0;
}