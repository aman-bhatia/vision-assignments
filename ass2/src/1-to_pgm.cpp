#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <dirent.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<string> read_directory(string path){
	vector<string> result;
	dirent* de;
	DIR* dp;
	errno = 0;
	dp = opendir(path.c_str());
	if(dp){
		while(true){
			errno = 0;
			de = readdir(dp);
			if (de == NULL) break;
			string filename = string(de->d_name);
			if (filename=="." or filename=="..") continue;
			result.push_back(filename);
		}
		closedir(dp);
	} else {
		cout << "Unable to open directory : " << path << endl << endl;
	}

	return result;
}

int main(int argc, char** argv){

	string path = "./dataset/jpeg/";
	vector<string> images = read_directory(path);

	vector<int> params;
	params.push_back(CV_IMWRITE_PXM_BINARY);
	params.push_back(1);

	for (int i=0;i<images.size();i++){
		Mat img = imread(path + images[i],0);
		imwrite("./dataset/pgm/" + images[i] + ".pgm",img, params);
	}
}
