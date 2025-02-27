//
// Created by Arya Bhattacharjee on 2/24/25.
//

#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <Eigen/Dense>
#include <vector>
#include <map>

class MnistReader {
private:
 int iMagic = 0;
 int iCount = 0;
 int iX = 0, iY = 0;

 int lMagic = 0;
 int lCount = 0;

 std::vector<std::tuple<Eigen::MatrixXd, u_char>> mnist;

public:
 MnistReader(int, std::string, std::string);
 void mnist_init(int, std::string, std::string);
 void shuffle(int);

 std::vector<std::tuple<Eigen::MatrixXd, u_char>> data() {
  return this->mnist;
 }

 u_char getLabel(int i) const{
  return std::get<1>(this->mnist[i]);
 }

 Eigen::MatrixXd& getImage(int i) {
  return std::get<0>(this->mnist[i]);
 }

 int image_size() const{
  return iX * iY;
 }

private:
 void readImages(std::ifstream&, int);
 void readLabels(std::ifstream&, int);
 static int binaryInt(int);
 static double scaledDouble(int, int);
};

#endif //MNIST_READER_H
