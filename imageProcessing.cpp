#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "imageProcessing.h"
#include <dirent.h>

using namespace std;
using namespace cv;

Mat loadGrayScaleImage(string imageFname) {
    Mat image;
    image = imread(imageFname, CV_LOAD_IMAGE_GRAYSCALE);

    if (!image.data) {
        throw runtime_error("Could not open or find the image");
    }

    return image;
}

void showImage(Mat image) {
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);

    waitKey(0);
    return;
}

vector<double> readImage(string imageFname) {
    Mat image = loadGrayScaleImage(imageFname);
    vector<double> dataElem;

    dataElem.assign(image.datastart, image.dataend);

    return dataElem;
}

vector<vector<double>> readImagesDir(string dirName) {
    cout << "Read data from dir " << dirName <<endl;

    vector<vector<double>> data;

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(dirName.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string imageFname = dirName + "/" + ent->d_name;
            try {
                data.push_back(readImage(imageFname));
            } catch (runtime_error err) {
//                cout << err.what() << endl;
            }
        }
        closedir(dir);
    }

    return data;
}

void readImagesData(string posDirName, string negDirName, vector<vector<double>> &data, vector<int> &labels) {
    vector<vector<double>> posData = readImagesDir(posDirName);
    data.insert(data.end(), posData.begin(), posData.end());
    labels.insert(labels.end(), posData.size(), 1);

    vector<vector<double>> negData = readImagesDir(negDirName);
    data.insert(data.end(), negData.begin(), negData.end());
    labels.insert(labels.end(), negData.size(), 0);
}

vector<double> get_mean_image(vector<string> &image_dirs) {
    cout << "Compute mean image" << endl;

    vector<double> mean_image;
    int num = 0;
    for (auto &dir : image_dirs) {
        vector<vector<double>> data = readImagesDir(dir);

        if (mean_image.empty()) {
            mean_image.insert(mean_image.end(), data[0].size(), 0.);
        }

        for (auto &data_elem : data) {
            for (size_t i = 0; i < data_elem.size(); ++i) {
                mean_image[i] += data_elem[i];
            }
            ++num;
        }
    }

    for (size_t i = 0; i < mean_image.size(); ++i) {
        mean_image[i] /= num;
    }
    return mean_image;
}


