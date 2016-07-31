#include <iostream>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


Mat warpFromParameters(Mat p){
	Mat warp(2,3,CV_64F);
	warp.at<double>(0,0) = 1 + p.at<double>(0,0);
	warp.at<double>(0,1) = p.at<double>(2,0);
	warp.at<double>(0,2) = p.at<double>(4,0);
	warp.at<double>(1,0) = p.at<double>(1,0);
	warp.at<double>(1,1) = 1 + p.at<double>(3,0);
	warp.at<double>(1,2) = p.at<double>(5,0);

	return warp;
}

// returns translational warp for the "window" in image "img1" in "img2"
Mat lktracker_trans(Mat img1, Mat img2, Rect window, Point2f init_point){
	
	Mat patch = img1(window);
	patch.convertTo(patch,CV_64F);
	img2.convertTo(img2,CV_64F);

	Mat kernel_x(1,3,CV_64F,Scalar(0.0));
	kernel_x.at<double>(0,0) = -0.5;		
	kernel_x.at<double>(0,2) = 0.5;		
	
	Mat kernel_y(3,1,CV_64F,Scalar(0.0));
	kernel_y.at<double>(0,0) = -0.5;
	kernel_y.at<double>(2,0) = 0.5;

	
	// two steepest descent images
	Mat dx,dy;
	filter2D(patch,dx,-1,kernel_x);
	filter2D(patch,dy,-1,kernel_y);

	vector<Mat> sd_images({dx,dy});


	// hessian calculation
	Mat H(2,2,CV_64F);
	for (int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			H.at<double>(i,j) = sum(sd_images[i].mul(sd_images[j]))[0];
		}
	}

	// hessian inverse
	Mat H_inv = H.inv();

	Mat p(6,1,CV_64F,Scalar(0.0));
	p.at<double>(4,0) -= init_point.x;
	p.at<double>(5,0) -= init_point.y;

	int iterations = 0;
	while(iterations++ < 50){

		Mat warp = warpFromParameters(p);
		
		// warp image		
		Mat warped_image;
		warpAffine(img2, warped_image, warp, patch.size());
		

		// error image
		Mat error_image(patch.size(),CV_64F);
		subtract(warped_image, patch, error_image, noArray(), CV_64F);

		
		// steepest descent parameter update
		Mat sd_update(2,1,CV_64F);
		for(int i=0;i<2;i++){
			sd_update.at<double>(i,0) = sum(sd_images[i].mul(error_image))[0];
		}

		// parameter update
		Mat delta_p = H_inv * sd_update;

		float del_norm = norm(delta_p);
		// cout << del_norm << endl;
		if (del_norm < 0.02){
			// cout << "CONVERGED\n\n";
			break;
		}
		p.at<double>(4,0) += delta_p.at<double>(0,0);
		p.at<double>(5,0) += delta_p.at<double>(1,0);
	}

	p.at<double>(4,0) += init_point.x;
	p.at<double>(5,0) += init_point.y;
	return warpFromParameters(p);
}


// returns affine warp for the "window" in image "img1" in "img2"
Mat lktracker_affine(Mat img1, Mat img2, Rect window, Point2f init_point){
	
	Mat patch = img1(window);
	patch.convertTo(patch,CV_64F);
	img2.convertTo(img2,CV_64F);

	Mat kernel_x(1,3,CV_64F,Scalar(0.0));
	kernel_x.at<double>(0,0) = -0.5;		
	kernel_x.at<double>(0,2) = 0.5;		
	
	Mat kernel_y(3,1,CV_64F,Scalar(0.0));
	kernel_y.at<double>(0,0) = -0.5;
	kernel_y.at<double>(2,0) = 0.5;

	
	// six steepest descent images
	Mat dx,dy;
	filter2D(patch,dx,-1,kernel_x);
	filter2D(patch,dy,-1,kernel_y);

	Mat xIx(patch.size(),CV_64F), yIx(patch.size(),CV_64F), xIy(patch.size(),CV_64F), yIy(patch.size(),CV_64F);
	for (int i=0;i<dx.rows;i++){
		for (int j=0;j<dx.cols;j++){
			xIx.at<double>(i,j) = j*dx.at<double>(i,j);
			xIy.at<double>(i,j) = j*dy.at<double>(i,j);
			yIx.at<double>(i,j) = i*dx.at<double>(i,j);
			yIy.at<double>(i,j) = i*dy.at<double>(i,j);
		}
	}
	vector<Mat> sd_images({xIx,xIy,yIx,yIy,dx,dy});


	// hessian calculation
	Mat H(6,6,CV_64F);
	for (int i=0;i<6;i++){
		for(int j=0;j<6;j++){
			H.at<double>(i,j) = sum(sd_images[i].mul(sd_images[j]))[0];
		}
	}

	// hessian inverse
	Mat H_inv = H.inv();

	Mat p(6,1,CV_64F,Scalar(0.0));
	p.at<double>(4,0) -= init_point.x;
	p.at<double>(5,0) -= init_point.y;

	int iterations = 0;
	while(iterations++ < 50){

		Mat warp = warpFromParameters(p);
		
		// warp image		
		Mat warped_image;
		warpAffine(img2, warped_image, warp, patch.size());
		

		// error image
		Mat error_image(patch.size(),CV_64F);
		subtract(warped_image, patch, error_image, noArray(), CV_64F);

		
		// steepest descent parameter update
		Mat sd_update(6,1,CV_64F);
		for(int i=0;i<6;i++){
			sd_update.at<double>(i,0) = sum(sd_images[i].mul(error_image))[0];
		}

		// parameter update
		Mat delta_p = H_inv * sd_update;

		float del_norm = norm(delta_p);
		// cout << del_norm << endl;
		if (del_norm < 0.02){
			// cout << "CONVERGED\n\n";
			break;
		}
		p.at<double>(0,0) += delta_p.at<double>(0,0) + p.at<double>(0,0)*delta_p.at<double>(0,0) + p.at<double>(2,0)*delta_p.at<double>(1,0);
		p.at<double>(1,0) += delta_p.at<double>(1,0) + p.at<double>(1,0)*delta_p.at<double>(0,0) + p.at<double>(3,0)*delta_p.at<double>(1,0);
		p.at<double>(2,0) += delta_p.at<double>(2,0) + p.at<double>(0,0)*delta_p.at<double>(2,0) + p.at<double>(2,0)*delta_p.at<double>(3,0);
		p.at<double>(3,0) += delta_p.at<double>(3,0) + p.at<double>(1,0)*delta_p.at<double>(2,0) + p.at<double>(3,0)*delta_p.at<double>(3,0);
		p.at<double>(4,0) += delta_p.at<double>(4,0) + p.at<double>(0,0)*delta_p.at<double>(4,0) + p.at<double>(2,0)*delta_p.at<double>(5,0);
		p.at<double>(5,0) += delta_p.at<double>(5,0) + p.at<double>(1,0)*delta_p.at<double>(4,0) + p.at<double>(3,0)*delta_p.at<double>(5,0);
	}

	p.at<double>(4,0) += init_point.x;
	p.at<double>(5,0) += init_point.y;
	return warpFromParameters(p);
}


Point2f pt(-1,-1);
static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/){
	if (event == EVENT_LBUTTONDOWN){
		pt = Point2f((float)x, (float)y);
	}
}


void track(char* video_path, string model){

	VideoCapture cap(video_path);
	assert(cap.isOpened());

	int max_frames = cap.get(CAP_PROP_FRAME_COUNT);

	Mat img1, img2, img1_gray, img2_gray;

	cap >> img1;
	cvtColor(img1, img1_gray, COLOR_BGR2GRAY);

	namedWindow("Choose point to track",1);
	setMouseCallback("Choose point to track", onMouse, 0);

	while(pt.x == -1){
		imshow("Choose point to track",img1_gray);
		char c = (char)waitKey(10);
		if(c == 27) break;
	}

	circle(img1_gray, pt, 6, Scalar(255), -1);
	imshow("Choose point to track",img1_gray);
	waitKey();
	
	while(true){

		int r = 60;
		Rect patch_rect(pt.x -r, pt.y -r, 2*r, 2*r);

		cap >> img2;
		if (img2.data==NULL) break;
		
		cvtColor(img2, img2_gray, COLOR_BGR2GRAY);

		Mat T;
		if (model == "trans"){
			T = lktracker_trans(img1_gray, img2_gray, patch_rect, Point2f(patch_rect.x, patch_rect.y));
			pt.x = (pt.x*T.at<double>(1,1)) + (-pt.y*T.at<double>(0,1)) - T.at<double>(0,2);
			pt.y = (-pt.x*T.at<double>(1,0)) + (pt.y*T.at<double>(0,0)) - T.at<double>(1,2);
		} else if (model == "affine"){
			T = lktracker_affine(img1_gray, img2_gray, patch_rect, Point2f(patch_rect.x, patch_rect.y));
			pt.x = (pt.x*T.at<double>(1,1)) + (-pt.y*T.at<double>(0,1)) - T.at<double>(0,2);
			pt.y = (-pt.x*T.at<double>(1,0)) + (pt.y*T.at<double>(0,0)) - T.at<double>(1,2);
		} else if (model == "opencv"){
			vector<Point2f> old_pt, new_pt;
			old_pt.push_back(pt);
			vector<uchar> status;
			vector<float> err;
			calcOpticalFlowPyrLK(img1_gray, img2_gray, old_pt, new_pt, status, err);
			if (!status[0]){
				cout << "Unable to track!\n\n";
				exit(0);
			}
			pt = new_pt[0];
		} else {
			cout << "model can only be one of affine,trans,opencv...\n\n";
			exit(0);
		}

		circle(img2_gray, pt, 6, Scalar(255), -1);
	
		imshow("Choose point to track",img2_gray);
		char c = (char)waitKey(30);
		if(c == 27) break;

		cvtColor( img2, img1_gray, COLOR_BGR2GRAY );
	}
}

int main(int argc, char** argv){
	if(argc < 3) {
		cout << "Usage : ./a.out <video path> <model(affine/trans/opencv)> \n\n";
		return 0;
	}
	string model = argv[2];
	track(argv[1],model);
}