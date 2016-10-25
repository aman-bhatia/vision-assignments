#include <iostream>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void timeToCollision(string path){

	VideoCapture cap(path);
	assert(cap.isOpened());

	Mat img1,img2,img1_gray,img2_gray, flow;

	int max_frames = cap.get(CAP_PROP_FRAME_COUNT);

	cap >> img1;
	cvtColor(img1, img1_gray, COLOR_BGR2GRAY);

	vector<float> times;
	while(true){

		cap >> img2;
		if (img2.data == NULL) break;	
		cvtColor(img2, img2_gray, COLOR_BGR2GRAY);

		calcOpticalFlowFarneback(img1_gray, img2_gray, flow, 0.5, 1, 30, 2, 7, 1.5, 0);

		// show optical flow
		for (int y = 0; y < img1.rows; y += 5){
			for (int x = 0; x < img1.cols; x += 5){
				Point2f flowatxy = flow.at<Point2f>(y, x);
				line(img2, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255,0,0));
				circle(img2, Point(x, y), 1, Scalar(0, 0, 0), -1);
			}
		}

		Mat uv[2];
		split(flow,uv);
		Mat ux, vy;
		Sobel( uv[0], ux, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
		Sobel( uv[1], vy, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);

		float div = 0;
		for (int i=0;i<ux.rows;i++){
			for (int j=0;j<ux.cols;j++){
				div += ux.at<float>(i,j) + vy.at<float>(i,j);
			}
		}
		if (div){
			times.push_back(2/div);
			cout << "time = " << 2/div << endl;
		}

		img1_gray = img2_gray.clone();
			
		imshow("OK",img2);
		char c = (char)waitKey(10);
		if(c == 27) break;
	}

	cout << "=========================\n\n";
	sort(times.begin(),times.end());

	// for (int i=0;i<times.size();i++){
	// 	cout << times[i]<< endl;
	// }
	cout << "Time to collision = " << times[times.size()/2] << endl;
}

int main(int argc, char** argv){
	timeToCollision(argv[1]);
}