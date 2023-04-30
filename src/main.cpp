/*
  Xiaowei Zhang
  S23
  
  read image file given an directory path, calculate feature vectors for each image inside the directory, and store them in a csv file
  calculate distance for each image given a image file
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "csv_util.h"
#include <sys/stat.h>

using namespace cv;
using namespace std;

const int HSIZE = 8;


//Task 1: Baseline Matching
//Feature vector: 9x9 square in the middle of the image
//Distance: Euclidean distance
void calDistBase(std::map<double, string>& order, vector<vector<string>> csvData, char* targetPath, int n){
  cv::Mat targetImg = cv::imread(targetPath, IMREAD_COLOR);
  std::vector<double> targetData;
  calFeatureVecBase(targetData, targetImg);
  for (int i=0; i<csvData.size(); i++){
    double diff = 0;
    for(int j=1; j<csvData.at(i).size(); j++){
      diff += (atoi(csvData.at(i).at(j).c_str()) - targetData.at(j-1))*(atoi(csvData.at(i).at(j).c_str()) - targetData.at(j-1));
    }
    order[diff] = csvData.at(i).at(0);
  }
    auto i=order.begin();
    while(i!=order.end() and n>0){
              std::cout << i->first
                  << " : "
                  << i->second << '\n';
                  i++;
                  n--;
    }
}
//Task 2: Color Histogram Matching
//Feature vector: color histogram
//Distance: intersection distance
void calIntersecDist(std::map<string, double>& order, vector<vector<string>> csvData, char* targetPath, const int hSize){
  cv::Mat targetImg = cv::imread(targetPath, IMREAD_COLOR);
  cv::Mat hist;
  std::vector<double> targetData;
  // if(isCustom){
  //   //  targetImg = targetImg(Rect(0, targetImg.rows/3, targetImg.cols, targetImg.rows*2/3));
  //   targetImg = preprocessImage(targetImg);
  //   //  cv::imshow("target", targetImg);
  //   //  waitKey(0);
  //   std::vector<float> hogFeatures;
  //   hogFeatures = computeHOGFeatures(targetImg);
  //   for(auto i : hogFeatures)
  //         targetData.push_back(i);
  // }
  // if(isCustom){
  //   Mat lower = targetImg(computeRectangle(targetImg));
  //   targetImg = lower;
  // }
  calHistogramVec(targetImg, hist, hSize, targetData);
  for (int i=0; i<csvData.size(); i++){
    double diff = 0;
    for(int j=1; j<csvData.at(i).size(); j++){
      double target = targetData[j-1];
      double curr = stod(csvData[i][j].c_str());
      if(curr < target){
        diff += curr;
      }
      else{
        diff += target;
      }
    }
    order[csvData.at(i).at(0)] = 1-diff;
  }
}
//Task3,4: Multi-histogram Matching and texture Matching
//Feature vector: upper half color histogram + lower half color histogram for multi-histogram matching and gradient magnitudes and whole color histogram for texture matching
//Distance: intersection distance
void calIntersecDist(std::map<string, double>& order, vector<vector<string>> csvData1, vector<vector<string>> csvData2, char* targetPath, const int hSize, int command){
  cv::Mat targetImg = cv::imread(targetPath, IMREAD_COLOR);
  int rows = targetImg.rows;
  int cols = targetImg.cols;
  std::vector<double> target1;
  std::vector<double> target2;
  if(command == 1){
    cv::Mat targetUpper = targetImg(Rect(0, 0, cols, rows/2));
    cv::Mat targetLower = targetImg(Rect(0, rows/2, cols, rows/2));
    cv::Mat hist;
    calHistogramVec(targetUpper, hist, hSize, target1);
    calHistogramVec(targetLower, hist, hSize, target2);
  }
  else if(command == 2){
    cv::Mat hist;
    calTextureVec(targetImg, hist, hSize, target1);
    calHistogramVec(targetImg, hist, hSize, target2);
  }

  for (int i=0; i<csvData1.size(); i++){
    double diff = 0;
    for(int j=1; j<csvData1.at(i).size(); j++){
      double target = target1[j-1];
      double curr = stod(csvData1[i][j].c_str());
      if(curr < target){
        diff += curr;
      }
      else{
        diff += target;
      }
    
    }
    order[csvData1.at(i).at(0)] = 0.5*(1-diff);
  }

  for (int i=0; i<csvData2.size(); i++){
    double diff = 0;
    for(int j=1; j<csvData2.at(i).size(); j++){
      double target = target2[j-1];
      double curr = stod(csvData2[i][j].c_str());
      if(curr < target){
        diff += curr;
      }
      else{
        diff += target;
      }
    }
    order[csvData2.at(i).at(0)] += 0.5*(1-diff);
  }
}
// Task 5
void calIntersecDist(std::map<string, double>& order, vector<vector<string>> csvData1, vector<vector<string>> csvData2, char* targetPath, const int hSize){
  cv::Mat targetImg = cv::imread(targetPath, IMREAD_COLOR);
  Mat lower = targetImg(computeRectangle(targetImg));
  targetImg = lower;
  int rows = targetImg.rows;
  int cols = targetImg.cols;
  std::vector<double> target1;
  std::vector<double> target2;

  cv::Mat hist;
  calHistogramVec(targetImg, hist, hSize, target1);
  
  target2 = gaborFeatures(targetImg);

  for (int i=0; i<csvData1.size(); i++){
    double diff = 0;
    for(int j=1; j<csvData1.at(i).size(); j++){
      double target = target1[j-1];
      double curr = stod(csvData1[i][j].c_str());
      if(curr < target){
        diff += curr;
      }
      else{
        diff += target;
      }
    
    }
    // order[1-diff] = csvData.at(i).at(0);
    order[csvData1.at(i).at(0)] = 0.7*(1-diff);
  }

  for (int i=0; i<csvData2.size(); i++){
    double diff = 0;
    for(int j=1; j<csvData2.at(i).size(); j++){
      double target = target2[j-1];
      double curr = stod(csvData2[i][j].c_str());
      if(curr < target){
        diff += curr;
      }
      else{
        diff += target;
      }
    }
    order[csvData2.at(i).at(0)] += 0.3*(1-diff);
  }
}

void readCsv(std::vector<std::vector<std::string>>& csvData, std::string fileName){
  //read the csv file and store the feature vectors of each image in a variable called csvData
    std::ifstream file (fileName);
    std::string line;
    if(!file.is_open()){
      std::cout<<"Cannot access "<<fileName<<" , please try again.";
      return;
    }
    
    while (std::getline(file, line)) {
        std::vector<std::string> fields;
        std::string picName;
        std::string field;
        std::istringstream lineStream(line);
        while (std::getline(lineStream, field, ',')) {
            fields.push_back(field);
        }
        csvData.push_back(fields);
    }
    file.close();

}

/*

  Input: diretory path, target path, n for top n matches
  Output:A csv file containing all feature vectors for each image in the directory 
 */
int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;

  // check for sufficient arguments
  if( argc < 2) {
    // printf("usage: %s <directory path>\n", argv[0]);
    printf("Error: unspecified directory path\n");
    exit(-1);
  }

  std::cout<<"Specify the task you want to perform"<<std::endl;
  std::cout<<"1 for Baseline Matching"<<std::endl;
  std::cout<<"2 for Histogram Matching"<<std::endl;
  std::cout<<"3 for Multi-Histogram Matching"<<std::endl;
  std::cout<<"4 for Texture and Color"<<std::endl;
  std::cout<<"5 for Custom Design"<<std::endl;
  // get the directory path, process each image, and store their feature vectors to csv file
  // This will only be executed once
  int in;
  cin>>in;
  const char* fileName;
  char targetPath[256];
  strcpy(targetPath, argv[2]);
  int n = atoi(argv[3]);
  if(directoryExists("vectorFiles") == false){
    mkdir("vectorFiles", 0777);
  }
  //Case 1: Baseline Matching
  if(in==1){
    std::vector<std::vector<std::string>> csvData;
    std::map<double, std::string> order;
    fileName = "vectorFiles/baseOutput.csv";
    std::ifstream f(fileName);
    if(!f.good()){
      f.close();
      strcpy(dirname, argv[1]);
      printf("Processing directory %s\n", dirname );
      readImageFiles(dirp, dirname, fileName, in);
    }
    readCsv(csvData, fileName);
    calDistBase(order, csvData, targetPath, n);
  }
  //Case 2: Histogram Matching
  else if(in == 2){
    std::vector<std::vector<std::string>> csvData;
    std::map<string, double> order;
    fileName = "vectorFiles/histogramOutput.csv";
    ifstream f(fileName);
    if(!f.good()){
      f.close();
      strcpy(dirname, argv[1]);
      printf("Processing directory %s\n", dirname);
      readImageFiles(dirp, dirname, fileName, in);
    }
    readCsv(csvData, fileName);
    calIntersecDist(order, csvData, targetPath, HSIZE);
    mapSort(order, n);
  }
  //Case 3: Multi-Histogram Matching
  else if(in == 3){
    fileName = "";
    char* upperFile = "vectorFiles/multiHistUpperOutput.csv";
    char* lowerFile = "vectorFiles/multiHistLowerOutput.csv";
    std::vector<std::vector<std::string>> csvData1;
    std::vector<std::vector<std::string>> csvData2;
    // ifstream f(fileName);
    ifstream f1(upperFile);
    ifstream f2(lowerFile);
    map<string, double> order;
    if(!f1.good() || !f2.good()){
      f1.close();
      f2.close();
      strcpy(dirname, argv[1]);
      printf("Processing directory %s\n", dirname);
      readImageFiles(dirp, dirname, fileName, in);
    }
    readCsv(csvData1, upperFile);
    readCsv(csvData2, lowerFile);
    calIntersecDist(order, csvData1, csvData2, targetPath, HSIZE, 1);
    mapSort(order, n);
  }
  //Case 4: Texture and Color
  else if(in == 4){
    std::vector<std::vector<std::string>> csvData1;
    std::vector<std::vector<std::string>> csvData2;
    std::map<string, double> order;
    char* colorFile = "vectorFiles/colorOutput.csv";
    char* textureFile = "vectorFiles/textureOutput.csv";
    ifstream f1(colorFile);
    ifstream f2(textureFile);
    if(!f1.good() || !f2.good()){
      f1.close();
      f2.close();
      strcpy(dirname, argv[1]);
      printf("Processing directory %s\n", dirname);
      readImageFiles(dirp, dirname, fileName, in);
    }
    readCsv(csvData1, textureFile);
    readCsv(csvData2, colorFile);
    calIntersecDist(order, csvData1, csvData2, targetPath, HSIZE, 2);
    mapSort(order, n);
    // calDist(order, csvData, targetPath, n);

  }
  //Case 5: Custom Design
  else if(in == 5)
  {
    std::vector<std::vector<std::string>> csvData1;
    std::vector<std::vector<std::string>> csvData2;
    std::map<string, double> order;
    char* colorFile = "vectorFiles/customColorOutput.csv";
    char* textureFile = "vectorFiles/customTextureOutput.csv";
    ifstream f1(colorFile);
    ifstream f2(textureFile);
    if(!f1.good() || !f2.good()){
      f1.close();
      f2.close();
      strcpy(dirname, argv[1]);
      printf("Processing directory %s\n", dirname);
      readImageFiles(dirp, dirname, fileName, in);
    }
    readCsv(csvData1, colorFile);
    readCsv(csvData2, textureFile);
    calIntersecDist(order, csvData1, csvData2, targetPath, HSIZE);
    mapSort(order, n);
  }
  else{
    return (1);
  }

  return(0);
}





