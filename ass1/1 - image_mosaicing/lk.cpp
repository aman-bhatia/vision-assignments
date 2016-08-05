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
		if (del_norm < 0.01){
			cout << "CONVERGED\n\n";
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
		if (del_norm < 0.01){
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

	p.at<double>(4,0) += init_point.x;
	p.at<double>(5,0) += init_point.y;
	return warpFromParameters(p);
}


void mosaic(Mat img1, Mat img2){
	// 4r is the height of the patch
	int r = (int)img1.rows/4;

	Rect patch_rect = Rect(img1.cols -1 - 2*r, img1.rows/2 - 2*r, 2*r,4*r);
	Mat patch = img1(patch_rect);
	// imshow("patch",patch);
	// waitKey();

	int i=0;
	int inc = 5;
	vector<Mat> temp;
	while(true){
		if ((i*inc + patch_rect.width) >= img1.cols) break;

		Mat warp = lktracker_affine(img1,img2,patch_rect,Point2f(i*inc,patch_rect.y));
		if (warp.rows==0 and warp.cols==0){
			i++;
			continue;
		}
		warp.at<double>(0,2) -= (i*inc);
		Mat warped_image;
		warpAffine(img2, warped_image, warp, img2.size());
		temp.push_back(warped_image);
		// imwrite(to_string(i) + ".jpg", warped_image);
		cout << "====================  " << i << endl;
		i++;
	}


	int min_error_index = 0;
	float min_error = INT_MAX;
	for (int i=0;i<temp.size();i++){
		Mat new_patch = temp[i](Rect(0 , patch_rect.y, patch_rect.width,patch_rect.height));
		Mat error_image(patch.size(), CV_64F);
		error_image = patch - new_patch;
		float error = norm(error_image);
		if (error < min_error){
			min_error = error;
			min_error_index = i;
		}
		cout << i << " => " << error << endl;
	}
	img2 = temp[min_error_index];

	// blending two images
	Mat img1_only = img1(Rect(0,0,patch_rect.x,img1.rows));
	Mat img2_only = img2(Rect(patch_rect.width,0,img2.cols - patch_rect.width, img2.rows));
	Mat img1_patch = img1(Rect(patch_rect.x,0,patch_rect.width,img1.rows));
	Mat img2_patch = img2(Rect(0,0,patch_rect.width,img2.rows));

	// imwrite("1only.jpg",img1_only);
	// imwrite("2only.jpg",img2_only);
	// imwrite("1patch.jpg",img1_patch);
	// imwrite("2patch.jpg",img2_patch);

	Mat result = Mat::zeros(img1.rows, img1_only.cols + img1_patch.cols + img2_only.cols, img1.type());

	img1_only.copyTo(result(Range::all(), Range(0, img1_only.cols)));
	img2_only.copyTo(result(Range::all(), Range(img1_only.cols + patch_rect.width, result.cols)));
	// imwrite("inter.jpg",result);

	for (int i=0;i<img1_patch.rows;i++){
		for(int j=0;j<img1_patch.cols;j++){
			float alpha = (float)j/img1_patch.cols;
			result.at<uchar>(i,img1_only.cols+j) = (1-alpha)*img1_patch.at<uchar>(i,j) + alpha*img2_patch.at<uchar>(i,j);
		}
	}

	imwrite("result.jpg",result);
}




int main(int argc, char** argv){
	if(argc < 3) {
		cout << "provide 2 images to create mosaic. Period.\n\n";
		return 0;
	}

	Mat img1, img2, img1_gray, img2_gray;
	img1 = imread(argv[1]);
	cvtColor( img1, img1_gray, COLOR_BGR2GRAY );

	img2 = imread(argv[2]);
	cvtColor( img2, img2_gray, COLOR_BGR2GRAY );

	mosaic(img1_gray, img2_gray);

}