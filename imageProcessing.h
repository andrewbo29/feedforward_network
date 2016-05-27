#include <opencv2/core/core.hpp>

using namespace cv;

Mat loadGrayScaleImage(string imageFname);

void showImage(Mat image);

vector<double> readImage(string imageFname);

vector<vector<double>> readImagesDir(string dirName);

void readImagesData(vector<string> dir_names, vector<vector<double>> &data, vector<vector<int>> &labels);

vector<double> get_mean_image(vector<string> &image_dirs);

Mat compute_mean_image(vector<string> &image_dirs);