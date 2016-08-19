#include <iostream>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int smoothing_window = 50;

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

// returns affine warp for the "window" in image "img1" in "img2"
Mat lktracker(Mat img1, Mat img2, Rect window){
	
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
	p.at<double>(4,0) -= window.x;
	p.at<double>(5,0) -= window.y;

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
		cout << del_norm << endl;
		if (del_norm < 0.02){
			cout << "CONVERGED\n\n";
			break;
		}
		p.at<double>(0,0) += delta_p.at<double>(0,0) + p.at<double>(0,0)*delta_p.at<double>(0,0) + p.at<double>(2,0)*delta_p.at<double>(1,0);
		p.at<double>(1,0) += delta_p.at<double>(1,0) + p.at<double>(1,0)*delta_p.at<double>(0,0) + p.at<double>(3,0)*delta_p.at<double>(1,0);
		p.at<double>(2,0) += delta_p.at<double>(2,0) + p.at<double>(0,0)*delta_p.at<double>(2,0) + p.at<double>(2,0)*delta_p.at<double>(3,0);
		p.at<double>(3,0) += delta_p.at<double>(3,0) + p.at<double>(1,0)*delta_p.at<double>(2,0) + p.at<double>(3,0)*delta_p.at<double>(3,0);
		p.at<double>(4,0) += delta_p.at<double>(4,0) + p.at<double>(0,0)*delta_p.at<double>(4,0) + p.at<double>(2,0)*delta_p.at<double>(5,0);
		p.at<double>(5,0) += delta_p.at<double>(5,0) + p.at<double>(1,0)*delta_p.at<double>(4,0) + p.at<double>(3,0)*delta_p.at<double>(5,0);
	}

	p.at<double>(4,0) += window.x;
	p.at<double>(5,0) += window.y;
	return warpFromParameters(p);
}


void Stablize(char* video_path){
	
	VideoCapture cap(video_path);
	assert(cap.isOpened());

	int max_frames = cap.get(CAP_PROP_FRAME_COUNT);

	Mat img1, img2, img1_gray, img2_gray;

	cap >> img1;
	cvtColor( img1, img1_gray, COLOR_BGR2GRAY );

	// chose a corner point to track
	Rect patch_rect(176-60,140-60,120,120);
	Mat jkl = img1(patch_rect);
	imshow("jkl",jkl);
	waitKey();

	// find displacement between frames

	vector<Point3f> displacement;

	int ith_frame = 0;
	while(true){
		cout << "frame - " << ith_frame << endl;
		ith_frame++;
		cap >> img2;
		if (img2.data==NULL) break;
		
		cvtColor( img2, img2_gray, COLOR_BGR2GRAY );

		Mat T = lktracker(img1_gray, img2_gray, patch_rect);
		
		cvtColor( img2, img1_gray, COLOR_BGR2GRAY );

		// decompose transform
		double dx = -T.at<double>(0,2);
		double dy = -T.at<double>(1,2);
		double da = -atan2(T.at<double>(1,0), T.at<double>(0,0));

		displacement.push_back(Point3f(dx, dy, da));
	}


	// find cumulative motion

	double x=0, y=0, a=0;
	vector<Point3f> motion;

	for(int i=0;i<displacement.size();i++){
		x += displacement[i].x;
		y += displacement[i].y;
		a += displacement[i].z;
		motion.push_back(Point3f(x,y,a));
	}

	// smoothen out

	vector<Point3f> smoothed_motion;

	for (int i=0;i<motion.size();i++){
		double new_x=0, new_y=0, new_a=0;
		int count=0;
		for (int j=-smoothing_window/2;j<=smoothing_window/2;j++){
			if ((i+j)>=0 and (i+j)<motion.size()){
				new_x += motion[i+j].x;
				new_y += motion[i+j].y;
				new_a += motion[i+j].z;
				count++;
			}
		}
		new_x = new_x/count;
		new_y = new_y/count;
		new_a = new_a/count;

		smoothed_motion.push_back(Point3f(new_x, new_y, new_a));
	}

	// calculate new displacement

	vector<Point3f> new_displacement;

	for(int i=0;i<displacement.size();i++){
		double dx = displacement[i].x + smoothed_motion[i].x - motion[i].x;
		double dy = displacement[i].y + smoothed_motion[i].y - motion[i].y;
		double da = displacement[i].z + smoothed_motion[i].z - motion[i].z;

		new_displacement.push_back(Point3f(dx, dy, da));
	}

	cap.set(CAP_PROP_POS_FRAMES,0);
	Mat T(2,3,CV_64F);

	Mat img, img_warped;
	VideoWriter out("stablized.mov", VideoWriter::fourcc('m','p', '4', 'v'), 30, img1.size());
	if(!out.isOpened()) {
		cout <<"Error! Unable to open video file for output.";
		exit(0);
	}

	for(int i=0;i<max_frames-1;i++){
		cap >> img;
		T.at<double>(0,0) = cos(new_displacement[i].z);
		T.at<double>(0,1) = -sin(new_displacement[i].z);
		T.at<double>(1,0) = sin(new_displacement[i].z);
		T.at<double>(1,1) = cos(new_displacement[i].z);
		T.at<double>(0,2) = new_displacement[i].x;
		T.at<double>(1,2) = new_displacement[i].y;

		warpAffine(img,img_warped,T,img.size());

		out << img_warped;

		// side by side
		Mat canvas = Mat::zeros(img.rows, img.cols*2+10, img.type());

		img.copyTo(canvas(Range::all(), Range(0, img.cols)));
		img_warped.copyTo(canvas(Range::all(), Range(img.cols+10, img.cols*2+10)));

		imshow("Video Stablization", canvas);
		waitKey(30);

	}
	out.release();
}



int main(int argc, char** argv){
	if(argc < 2) {
		cout << "provide video location\n\n";
		return 0;
	}

	Stablize(argv[1]);
}