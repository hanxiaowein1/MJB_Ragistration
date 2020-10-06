#pragma once

#ifndef _REGISTRATION_H_
#define _REGISTRATION_H_

#include <iostream>
#include <vector>
#include "TaskThread.h"
#include "PopQueueData.h"
#include "GetPhoneData.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "SlideFactory.h"
#include "SlideRead.h"

class Registration : public TaskThread
{
private:
	std::string m_slide_path;
	double m_slide_mpp = 0.0f;
	int m_slide_ratio = 0;
	std::string m_dst_path;
	//std::mutex read_mutex;
	//std::unique_ptr<SlideRead> read_handle;
	std::vector<std::mutex> read_mutexs;
	std::vector<std::unique_ptr<SlideRead>> read_handles;
	std::atomic<int> m_write_num = 0;

	//通过类内部队列来传输数据
	std::mutex data_mutex;
	std::queue<cv::Mat> data_queue;

	//template match gpu不支持多线程，那么就设置一个锁，同时只能让一个线程执行matchtemplate
	std::mutex match_mutex;

	//cv::Mat m_phone_img;
	cv::Mat m_level_img;
	cv::cuda::GpuMat m_gpu_level_img;
	int m_read_level = 4;
	double m_phone_img_mpp = 0.4284f;

	double m_threshold_1 = 0.5f;
	double m_threshold_2 = 0.85f;

	double max_angle = 1.0f;
	double max_scale = 0.04f;

	std::atomic<bool> m_debug_flag = false;

private:
	std::vector<std::pair<double, double>> getRotateScaleSet(
		int rotate_num, int scale_num, double angle, double scale, double angle_center = 0.0f, double scale_center = 1.0f);
	//将一张图像旋转缩放一定次数
	void getRotateScaleImage(
		cv::Mat& src, std::vector<cv::Mat>& dst_s, 
		std::vector<std::pair<double, double>> rotate_scale_set);
	void getRotateScaleImage(
		cv::Mat& src, std::vector<cv::Mat>& dst_s, 
		int rotate_num, int scale_num, double angle, double scale, double angle_center = 0.0f, double scale_center = 1.0f);
	void getRotateScaleImage(
		cv::cuda::GpuMat& src, std::vector<cv::cuda::GpuMat>& dst_s, 
		std::vector<std::pair<double, double>> rotate_scale_set);
	void getRotateScaleImage(
		cv::cuda::GpuMat& src, std::vector<cv::cuda::GpuMat>& dst_s, 
		int rotate_num, int scale_num, double angle, double scale, double angle_center = 0.0f, double scale_center = 1.0f);
	int findBestLocVal(
		cv::cuda::GpuMat& adjust_img, cv::cuda::GpuMat& standard_img, 
		std::vector<std::pair<double, double>> rotate_scale_set, std::vector<std::pair<cv::Point, double>>& ret_result);
	int findBestLocVal(
		cv::Mat& adjust_img, cv::Mat& standard_img,
		std::vector<std::pair<double, double>> rotate_scale_set, std::vector<std::pair<cv::Point, double>>& ret_result);
	//在bigImg中寻找templateImg的最佳模板匹配
	std::pair<cv::Point, double> findMatchLocAndValue(cv::Mat& template_img, cv::Mat& big_img);
	std::pair<cv::Point, double> findMatchLocAndValue(cv::cuda::GpuMat& gpu_template_img, cv::cuda::GpuMat& gpu_big_img);
	//获取最大的分数的位置
	int getBiggestLocValue(std::vector<std::pair<cv::Point, double>>& loc_vals);
	cv::Rect getRotateImgCenterRect(cv::Mat& img, double angle);
	cv::Rect getRotateImgCenterRect(cv::cuda::GpuMat& img, double angle);
	cv::Rect getCenterRect(int x0, int y0, double angle);
	//直接寻找手机图像拍摄的匹配图像
	bool matchPhoneBigImage(cv::Mat& phone_img, cv::Mat& slide_img, cv::Mat& together_img, std::pair<cv::Point, double>& ret_result);
	//在big_img中寻找small_img子图的最佳匹配（small_img是big_img中的一个子图）
	bool findFinalImg(cv::Mat& small_img, cv::Mat& big_img, cv::Mat &together_img, std::pair<cv::Point, double> &ret_result);
	std::vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width, int overlap, bool flag_right = false, bool flag_down = false);
	void initL4Img();
	void initReadHandle();
	//手机图像的四个方向校准
	std::pair<cv::Point, double> adjustPhoneImgOri(cv::Mat& phone_img);
	void getSlideTile(int read_handle_index, int read_level, cv::Rect rect, cv::Mat& read_img);
	void popData(cv::Mat& img);
	//得到手机图像以及其对应的切片图像
	cv::Point getMatchedSlideImage(cv::Mat& phone_img, cv::Mat& slide_img, int read_handle_index);
	std::pair<float, float> getRatios();
public:
	Registration();
	~Registration();
	void loadPhoneImg(std::string path);

	void process_single_data(int i);
	void process_single_data2(int i);
	void process(std::string phone_img_path, std::string sdpc_path, std::string dst_path);
	void saveImg(std::string savePath);
};

#endif