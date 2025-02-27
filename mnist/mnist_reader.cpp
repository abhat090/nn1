//
// Created by Arya Bhattacharjee on 2/24/25.
//

#include "mnist_reader.h"

#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <Eigen/Dense>

#define B_CONST 4
#define MAX_T_SIZE 60000
#define D_SCALAR 255

using namespace std;

MnistReader::MnistReader(int t_size, string path_i, string path_l) {
    assert(t_size <= MAX_T_SIZE);
    this->mnist = vector<tuple<Eigen::MatrixXd, u_char>>(t_size);
    mnist_init(t_size, path_i, path_l);
}



void  MnistReader::mnist_init(int t_size, string path_i, string path_l) {
    const string path = filesystem::current_path().generic_string();

    ifstream fileImage(path+path_i, ios::binary);
    ifstream fileLabel(path+path_l, ios::binary);

    if(!fileImage.is_open()) return;
    if(!fileLabel.is_open()) return;

    // Image Parameters
    fileImage.read((char *)&iMagic,sizeof(int));
    fileImage.read((char *)&iCount, sizeof(int));

    fileImage.read((char *)&iX, sizeof(int));
    fileImage.read((char *)&iY, sizeof(int));

    iMagic = binaryInt(iMagic);
    iCount = binaryInt(iCount);
    iX = binaryInt(iX);
    iY = binaryInt(iY);

    // Label Parameters
    fileLabel.read((char *)&lMagic, sizeof(int));
    fileLabel.read((char *)&lCount, sizeof(int));

    lMagic = binaryInt(lMagic);
    lCount = binaryInt(lCount);

    readImages(fileImage, t_size);
    readLabels(fileLabel, t_size);

    cout << "IMagic: " << iMagic << endl;
    cout << "Image Count: " << iCount << endl;
    cout << "X: " << iX << " ,Y: " << iY << endl;
    cout << "LMagic: " << lMagic << endl;
    cout << "Label Count: " << lCount << endl;
}

void MnistReader::shuffle(int seed) {
    std::shuffle(this->mnist.begin(), this->mnist.end(), default_random_engine(seed));
}


void MnistReader::readImages(std::ifstream& file, int t_size) {
    for(int i = 0; i < t_size; i++) {
        std::get<0>(this->mnist[i]) = Eigen::MatrixXd(iX, iY);
        std::get<0>(this->mnist[i]).fill(0);
        for(int x = 0; x < iX; x++) {
            for(int y = 0; y < iY; y++) {
                unsigned char temp;
                file.read((char *)&temp, sizeof(unsigned char));

                std::get<0>(this->mnist[i])(x, y) = scaledDouble((int)temp, D_SCALAR);
            }
        }
    }
}

void MnistReader::readLabels(std::ifstream& file, int t_size) {
    for(int i = 0; i < t_size; i++) {
        unsigned char temp;
        file.read((char *)&temp, sizeof(unsigned char));
        std::get<1>(this->mnist[i]) = temp;
    }
}

int MnistReader::binaryInt(int i) {
    unsigned char val[B_CONST];
    int final = 0;

    for (int j = 0; j < B_CONST; j++) {
        val[j] = (i >> (j * 8)) & 255;
    }

    for (int k = 0; k < B_CONST; k++) {
        final += (int) val[k] << (B_CONST-k-1) * 8;
    }

    return final;
}

double MnistReader::scaledDouble(int i, int scalar) {
    return (double)i/(double)scalar;
}
