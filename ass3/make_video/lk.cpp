#include <iostream>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main(int argc, char** argv){

	Size sz(400,400);
	VideoWriter out("high_speed.mov", VideoWriter::fourcc('m','p', '4', 'v'), 30, sz);
	if(!out.isOpened()) {
		cout <<"Error! Unable to open video file for output.";
		exit(0);
	}

	Mat img = Mat::zeros(sz,CV_8UC3);
	
	int k=0;
	while(k++ < 100){
		for (int i=200-2*k;i<=200+2*k;i++){
			for (int j=200-2*k;j<=200+2*k;j++)
				img.at<Vec3b>(i,j) = {0,255,0};
		}
		out << img;
	}

	out.release();

}