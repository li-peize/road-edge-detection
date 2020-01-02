#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

cv::Mat ero_dila(cv::Mat img, int iter1, int iter2) {
        cv::Mat ero_dila_img;
        cv::erode(img, ero_dila_img, Mat(), Point(-1,-1), iter1);
        cv::dilate(ero_dila_img, ero_dila_img, Mat(), Point(-1,-1), iter2);
        return ero_dila_img;
}

int main(int argc, char** argv)
{
        // Declare the output variables
        // Mat dst, cdst, cdstP;
        // const char* default_file = "sudoku.png";
        // const char* filename = argc >=2 ? argv[1] : default_file;

        // Loads an image
        cv::Mat src = cv::imread( "2.jpg", IMREAD_GRAYSCALE );
        cv::Mat img = cv::imread( "2.jpg" );


        // Check if image is loaded fine
        if(src.empty()) {
                printf(" Error opening image\n");
                return -1;
        }


////////////////////////////////////////////////////////////////////////////////
// BRIGHTNESS & CONTRAST
        Mat new_image = Mat::zeros( src.size(), src.type() );
        float alpha = 2.1; /*< Simple contrast control */
        int beta = 1.4; /*< Simple brightness control */
        // MEAN
        int mean = 0;
        for( int y = 0; y < src.rows; y++ ) {
                for( int x = 0; x < src.cols; x++ ) {
                        mean += src.at<uchar>(y,x);
                }
        }
        mean = mean / src.rows / src.cols;
        // BR & CT
        for( int y = 0; y < src.rows; y++ ) {
                for( int x = 0; x < src.cols; x++ ) {
                        new_image.at<uchar>(y,x) =
                                saturate_cast<uchar>( alpha*(src.at<uchar>(y,x)-mean) + beta*mean );
                }
        }

////////////////////////////////////////////////////////////////////////////////
// GAUSSIAN BLUR
        cv::Mat blur_img;
        cv::GaussianBlur( new_image, blur_img, Size( 31, 21 ), 0, 0 );

////////////////////////////////////////////////////////////////////////////////
// BINARY
        cv::Mat binary_img;
        cv::threshold(blur_img, binary_img, 200, 255, cv::THRESH_BINARY);

////////////////////////////////////////////////////////////////////////////////
// EROSION & DILATION

        cv::Mat ero_img = ero_dila(binary_img, 12, 0);
        cv::Mat dila_img = ero_dila(ero_img, 0, 20);
        cv::Mat ero_img_2 = ero_dila(dila_img, 5, 0);

////////////////////////////////////////////////////////////////////////////////
// AND
        cv::Mat and_img;
        cv::bitwise_and(dila_img, ~ero_img_2, and_img);

////////////////////////////////////////////////////////////////////////////////
// HOUGH
        std::vector<Vec4i> lines;
        cv::HoughLinesP(and_img, lines, 2, CV_PI/60, 160, 200, 25);
        for (size_t i = 0; i < lines.size(); i++) {
                cv::Vec4i line = lines[i];
                cv::line(img, cv::Point(line[0],line[1]), cv::Point(line[2],line[3]), Scalar(255,0,0), 5, cv::LINE_AA);
        }

////////////////////////////////////////////////////////////////////////////////
// SHOW
        Mat dst;
        resize(img, dst, Size(), 0.3, 0.3);
        imshow("Source", dst);
        // imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
        // imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);

        // Wait and Exit
        waitKey();
        return 0;
}
