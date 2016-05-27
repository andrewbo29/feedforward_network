#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "imageProcessing.h"
#include <dirent.h>
#include <opencv2/imgproc/imgproc.hpp>

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

void readImagesData(vector<string> dir_names, vector<vector<double>> &data, vector<vector<int>>&labels) {
    for (size_t i = 0; i < dir_names.size(); ++i) {
        vector<vector<double>> read_data = readImagesDir(dir_names[i]);
        data.insert(data.end(), read_data.begin(), read_data.end());
        vector<int> label(dir_names.size());
        std::fill(label.begin(), label.end(), 0);
        label[i] = 1;
        labels.insert(labels.end(), read_data.size(), label);
    }
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

Mat compute_mean_image(vector<string> &image_dirs) {
    int sum_row = 0;
    int sum_col = 0;
    int num = 0;

    for (auto &dirName : image_dirs) {
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(dirName.c_str())) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                string imageFname = dirName + "/" + ent->d_name;
                try {
                    Mat image = loadGrayScaleImage(imageFname);
                    sum_row += image.rows;
                    sum_col += image.cols;
                    ++num;
                } catch (runtime_error err) {
//                cout << err.what() << endl;
                }
            }
            closedir(dir);
        }
    }

    int width = sum_col / num;
    int height = sum_row / num;

    cv::Mat avgImg;
    avgImg.create(width, height, CV_32F);

    for (auto &dirName : image_dirs) {
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(dirName.c_str())) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                string imageFname = dirName + "/" + ent->d_name;
                try {
                    Mat image = loadGrayScaleImage(imageFname);
                    cv::accumulate(image, avgImg);
                } catch (runtime_error err) {
//                cout << err.what() << endl;
                }
            }
            closedir(dir);
        }
    }

//    for(i = 1; i <= N; i++){
//        image = imread(fileName.c_str(),0);
//        cv::accumulate(image, avgImg);
//    }

    avgImg = avgImg / num;
    return avgImg;
}




