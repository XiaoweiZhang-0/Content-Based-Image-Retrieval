/*
Xiaowei Zhang

CS 5330 Computer Vision
Spring 2023

CPP functions for reading CSV files with a specific format
- first column is a string containing a filename or path
- every other column is a number

The function returns a std::vector of char* for the filenames and a 2D std::vector of floats for the data
*/

#include <cstdio>
#include <iostream>
#include <cstring>
#include <vector>
#include "opencv2/opencv.hpp"
#include <cstring>
#include <dirent.h>
#include <fstream>

using namespace std;
using namespace cv;

const int HSIZE = 8;
/*
  reads a string from a CSV file. the 0-terminated string is returned in the char array os.

  The function returns false if it is successfully read. It returns true if it reaches the end of the line or the file.
 */
int getstring( FILE *fp, char os[] ) {
  int p = 0;
  int eol = 0;
  
  for(;;) {
    char ch = fgetc( fp );
    if( ch == ',' ) {
      break;
    }
    else if( ch == '\n' || ch == EOF ) {
      eol = 1;
      break;
    }
    // printf("%c", ch ); // uncomment for debugging
    os[p] = ch;
    p++;
  }
  // printf("\n"); // uncomment for debugging
  os[p] = '\0';

  return(eol); // return true if eol
}

int getint(FILE *fp, int *v) {
  char s[256];
  int p = 0;
  int eol = 0;

  for(;;) {
    char ch = fgetc( fp );
    if( ch == ',') {
      break;
    }
    else if(ch == '\n' || ch == EOF) {
      eol = 1;
      break;
    }
      
    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atoi(s);

  return(eol); // return true if eol
}

/*
  Utility function for reading one float value from a CSV file

  The value is stored in the v parameter

  The function returns true if it reaches the end of a line or the file
 */
int getfloat(FILE *fp, float *v) {
  char s[256];
  int p = 0;
  int eol = 0;

  for(;;) {
    char ch = fgetc( fp );
    if( ch == ',') {
      break;
    }
    else if(ch == '\n' || ch == EOF) {
      eol = 1;
      break;
    }
      
    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atof(s);

  return(eol); // return true if eol
}

/*
  Given a filename, and image filename, and the image features, by
  default the function will append a line of data to the CSV format
  file.  If reset_file is true, then it will open the file in 'write'
  mode and clear the existing contents.

  The image filename is written to the first position in the row of
  data. The values in image_data are all written to the file as
  floats.

  The function returns a non-zero value in case of an error.
 */
int append_image_data_csv( char *filename, char *image_filename, std::vector<double> &image_data, int reset_file ) {
  char buffer[256];
  char mode[8];
  FILE *fp;

  strcpy(mode, "a");

  if( reset_file ) {
    strcpy( mode, "w" );
  }
  
  fp = fopen( filename, mode );
  if(!fp) {
    printf("Unable to open output file %s\n", filename );
    exit(-1);
  }

  // write the filename and the feature vector to the CSV file
  strcpy(buffer, image_filename);
  std::fwrite(buffer, sizeof(char), strlen(buffer), fp );
  // std::cout<<(image_data)<<std::endl;
  for(int i=0;i<image_data.size();i++) {
    char tmp[256];
    sprintf(tmp, ",%.4f", image_data[i] );
    std::fwrite(tmp, sizeof(char), strlen(tmp), fp );
  }
      
  std::fwrite("\n", sizeof(char), 1, fp); // EOL

  fclose(fp);
  
  return(0);
}

/*
  Given a file with the format of a string as the first column and
  floating point numbers as the remaining columns, this function
  returns the filenames as a std::vector of character arrays, and the
  remaining data as a 2D std::vector<float>.

  filenames will contain all of the image file names.
  data will contain the features calculated from each image.

  If echo_file is true, it prints out the contents of the file as read
  into memory.

  The function returns a non-zero value if something goes wrong.
 */
int read_image_data_csv( char *filename, std::vector<char *> &filenames, std::vector<std::vector<float>> &data, int echo_file ) {
  FILE *fp;
  float fval;
  char img_file[256];

  fp = fopen(filename, "r");
  if( !fp ) {
    printf("Unable to open feature file\n");
    return(-1);
  }

  printf("Reading %s\n", filename);
  for(;;) {
    std::vector<float> dvec;
    
    
    // read the filename
    if( getstring( fp, img_file ) ) {
      break;
    }
    // printf("Evaluting %s\n", filename);

    // read the whole feature file into memory
    for(;;) {
      // get next feature
      float eol = getfloat( fp, &fval );
      dvec.push_back( fval );
      if( eol ) break;
    }
    // printf("read %lu features\n", dvec.size() );

    data.push_back(dvec);

    char *fname = new char[strlen(img_file)+1];
    strcpy(fname, img_file);
    filenames.push_back( fname );
  }
  fclose(fp);
  printf("Finished reading CSV file\n");

  if(echo_file) {
    for(int i=0;i<data.size();i++) {
      for(int j=0;j<data[i].size();j++) {
	printf("%.4f  ", data[i][j] );
      }
      printf("\n");
    }
    printf("\n");
  }

  return(0);
}
//Task 1 Feature calculation
void calFeatureVecBase(std::vector<double> &vec, cv::Mat img){
      int x = (img.cols - 9)/2+1;
      int y = (img.rows - 9)/2+1;
      cv::Mat subImg = img(cv::Rect(x, y, 9, 9));
      cv::Mat mat = subImg;
      for (int i = 0; i < mat.rows; ++i) {
            vec.insert(vec.end(), mat.ptr<uchar>(i), mat.ptr<uchar>(i) + mat.cols * mat.elemSize());
      }
}

//Task 2,3,5 Feature calculation
void calHistogramVec(cv::Mat& src, cv::Mat& hist, const int hSize, std::vector<double>& vec){
  const int divisor = 256 / hSize;
  int i, j, k;
  const int dim[3] = {hSize, hSize, hSize};
  hist = cv::Mat::zeros(3, dim, CV_64F);

  double sum = 0;
  for(i=0; i<src.rows; i++){
    Vec3b *ptr = src.ptr<Vec3b>(i);
    for(j=0; j<src.cols; j++){
      int b = ptr[j][0] / divisor;
      int g = ptr[j][1] / divisor;
      int r = ptr[j][2] / divisor;
      hist.at<double>(r, g, b)++; 
      sum ++;
    }
  }

  for(i=0; i<hSize;i++){
    for(j=0; j<hSize; j++){
      for(k=0; k<hSize; k++){
        hist.at<double>(i,j,k) /= sum;
        vec.insert(vec.end(), hist.at<double>(i, j, k));
      }
    }
  }
  

}

//Task 4 Texture Feature calculation
void calTextureVec(cv::Mat& src, cv::Mat& hist, const int hSize, std::vector<double>& vec){
    
    // Turn image into grayscale.
    cv::Mat grayscaleImage;
    cv::cvtColor(src, grayscaleImage, cv::COLOR_BGR2GRAY);

    cv::Mat sobelX, sobelY, sobelMagnitude;

    // Apply the Sobel X filter.
    cv::Sobel(grayscaleImage, sobelX, CV_32F, 1, 0);

    // Apply the Sobel Y filter.
    cv::Sobel(grayscaleImage, sobelY, CV_32F, 0, 1);

    // Calculate the Sobel magnitude image.
    cv::magnitude(sobelX, sobelY, sobelMagnitude);


    const int divisor = 256 / hSize;
    int i, j, k;
    const int dim[3] = {hSize, hSize, hSize};
    hist = cv::Mat::zeros(3, dim, CV_64F);


    double sum = 0;
    for(i=0; i<sobelMagnitude.rows; i++){
      Vec3b *ptr = sobelMagnitude.ptr<Vec3b>(i);
      for(j=0; j<sobelMagnitude.cols; j++){
        int b = ptr[j][0] / divisor;
        int g = ptr[j][1] / divisor;
        int r = ptr[j][2] / divisor;
        hist.at<double>(r, g, b)++; 
        sum ++;
      }
    }
    
    //normalize the histogram
    for(i=0; i<hSize;i++){
      for(j=0; j<hSize; j++){
        for(k=0; k<hSize; k++){
          hist.at<double>(i,j,k) /= sum;
          vec.insert(vec.end(), hist.at<double>(i, j, k));
        }
      }
    }
  

}
std::vector<double> gaborFeatures(const cv::Mat& image)
{
    // Convert the input image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Define Gabor filter parameters
    std::vector<int> kernelSizes = {21, 21}; // You can adjust kernel sizes based on your image dimensions
    std::vector<double> sigmas = {5, 5}; // Standard deviation of the Gaussian envelope
    std::vector<double> lambdas = {10, 10}; // Wavelength of the sinusoidal factor
    std::vector<double> gammas = {0.5, 0.5}; // Spatial aspect ratio
    std::vector<double> phis = {0, CV_PI / 2}; // Phase offset

    std::vector<double> features;

    // Apply Gabor filters with different scales and orientations
    for (size_t i = 0; i < kernelSizes.size(); ++i)
    {
        for (size_t j = 0; j < phis.size(); ++j)
        {
            cv::Mat kernel = cv::getGaborKernel(
                cv::Size(kernelSizes[i], kernelSizes[i]), sigmas[i], phis[j],
                lambdas[i], gammas[i], 0, CV_32F);

            cv::Mat filtered;
            cv::filter2D(grayImage, filtered, CV_32F, kernel);

            // Calculate the mean and standard deviation of the filtered response
            cv::Scalar mean, stddev;
            cv::meanStdDev(filtered, mean, stddev);

            features.push_back(mean[0]);
            features.push_back(stddev[0]);
        }
    }
    // Normalize the feature vector using L2-norm (Euclidean norm)
    int sum = 0;
    for(int i=0; i<features.size(); i++){
      sum += features[i];
    }
    for(int i=0; i<features.size(); i++){
     features[i] /= sum;
    }
    return features;
}



Rect computeRectangle(const Mat& image) {
    int imageWidth = image.cols;
    int imageHeight = image.rows;

    int lowerRectWidth = imageWidth * 2 / 3;
    int lowerRectHeight = imageHeight * 2 / 3;

    int lowerRectX = (imageWidth - lowerRectWidth) / 2;
    int lowerRectY = imageHeight - lowerRectHeight;

    Rect lowerMiddleRect(lowerRectX, lowerRectY, lowerRectWidth, lowerRectHeight);

    return lowerMiddleRect;
}


//read image files and store them as vectors into a csv file
int readImageFiles(DIR *dirp, char* dirname, const char* fileName, int taskNum){
    struct dirent *dp;
    // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing and calculate the feature vector for each image
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
    strstr(dp->d_name, ".png") ||
    strstr(dp->d_name, ".ppm") ||
    strstr(dp->d_name, ".tif") ) {
      string dir(dirname);
      string image(dp->d_name);
      string ans = dir + "/"+ image;
      Mat img = cv::imread(ans, IMREAD_COLOR);
      std::vector<double> vec;
      if(taskNum == 1){
        calFeatureVecBase(vec, img);
        append_image_data_csv((char*)fileName, dp->d_name, vec, 0);
      }
      else if(taskNum == 2){
        cv::Mat hist;
        calHistogramVec(img, hist, HSIZE, vec);
        append_image_data_csv((char*)fileName, dp->d_name, vec, 0);
      }
      else if(taskNum == 3){
        cv::Mat hist1;
        cv::Mat hist2;
        std::vector<double> vec1;
        std::vector<double> vec2;
        Mat upper = img(Rect(0, 0, img.cols, img.rows/2));
        Mat lower = img(Rect(0, img.rows/2, img.cols, img.rows/2));
        calHistogramVec(upper, hist1, HSIZE, vec1);
        string upperFileName = "vectorFiles/multiHistUpperOutput.csv";
        string lowerFileName = "vectorFiles/multiHistLowerOutput.csv";
        append_image_data_csv((char*)upperFileName.c_str(), dp->d_name, vec1, 0);
        calHistogramVec(lower, hist2, HSIZE, vec2);
        append_image_data_csv((char*)lowerFileName.c_str(), dp->d_name, vec2, 0);
      }
      else if(taskNum == 4){
        cv::Mat colorHist;
        cv::Mat textHist;

        //calculate color histogram and store it in colorOutput.csv
        std::vector<double> colorVec;
        calHistogramVec(img, colorHist, HSIZE, colorVec);
        char* colorFileName = "vectorFiles/colorOutput.csv";
        append_image_data_csv((char*)colorFileName, dp->d_name, colorVec, 0);
        
        
        
        std::vector<double> textVec;
        calTextureVec(img, textHist, HSIZE, textVec);
        //calculate text histogram and store it in textOutput.csv
        char* textFileName = "vectorFiles/textureOutput.csv";
        append_image_data_csv((char*)textFileName, dp->d_name, textVec, 0);

      }
      //this is for custom design
      else if(taskNum == 5){
        cv::Mat hist1;
        std::vector<double> vec1;
        string fileName1 = "vectorFiles/customColorOutput.csv";
        // calHistogramVec(lower, hist1, 8, vec1);
        // vec1 = computeColorFeatures(img, 8, 8, 8);

        Mat lower = img(computeRectangle(img));
        calHistogramVec(lower, hist1, 8, vec1);
        append_image_data_csv((char*)fileName1.c_str(), dp->d_name, vec1, 0);

        string fileName2 = "vectorFiles/customTextureOutput.csv";
        std::vector<double> vec2 = gaborFeatures(lower);
        append_image_data_csv((char*)fileName2.c_str(), dp->d_name, vec2, 0);

      }
    
    }

  }
  
  printf("Terminating\n");
  return 0;
}


// Comparator function to sort pairs
// according to second value
bool cmp(std::pair<std::string, double>& a,
        std::pair<std::string, double>& b)
{
    return a.second < b.second;
}

// Function to sort the map according
// to value in a (key-value) pairs
void mapSort(map<string, double>& M, int n)
{
 
    // Declare vector of pairs
    vector<pair<string, double> > A;
 
    // Copy key-value pair from Map
    // to vector of pairs
    for (auto& it : M) {
        A.push_back(it);
    }
 
    // Sort using comparator function
    sort(A.begin(), A.end(), cmp);
 
    // Print the sorted value
    auto i=A.begin();
    while(i!=A.end() and n>0){
              std::cout << i->first
                  << " : "
                  << i->second << '\n';
                  i++;
                  n--;
    }
}

//helper function to check if a folder exists
bool directoryExists(const std::string& directoryPath)
{
    // Check if the path exists and is a directory.
    return std::__fs::filesystem::exists(directoryPath) && std::__fs::filesystem::is_directory(directoryPath);
}


