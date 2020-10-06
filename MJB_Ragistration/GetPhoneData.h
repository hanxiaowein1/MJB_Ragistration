#pragma once
#ifndef _AILAB_GETPHONEDATA_H_
#define _AILAB_GETPHONEDATA_H_
#include "PopQueueData.h"
#include "opencv2/opencv.hpp"

class GetPhoneData : public PopQueueData<cv::Mat>
{
private:
	std::string m_img_path;//手机图像所在路径
public:
	void readImg(std::string img_path);
	void process(std::string imgs_path, std::string suffix = "jpg");
};

#endif