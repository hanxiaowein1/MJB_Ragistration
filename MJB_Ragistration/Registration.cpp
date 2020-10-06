#include "Registration.h"

using namespace std;

Registration::Registration()
{
}

Registration::~Registration()
{
}

std::vector<std::pair<double, double>> Registration::getRotateScaleSet(
	int rotate_num, int scale_num, double angle, double scale, double angle_center, double scale_center)
{
	double min_rotate_angle = angle_center - (rotate_num / 2) * angle;
	double min_scale = scale_center - (scale_num / 2) * scale;
	std::vector<std::pair<double, double>> rotate_scale_set;
	for (int i = 0; i <= rotate_num; i++)
	{
		std::pair<double, double> single_elem;
		double temp_rotate_angle = min_rotate_angle + i * angle;
		for (int j = 0; j <= scale_num; j++)
		{
			double temp_scale = min_scale + j * scale;
			single_elem.first = temp_rotate_angle;
			single_elem.second = temp_scale;
			rotate_scale_set.emplace_back(single_elem);
		}
	}
	return rotate_scale_set;
}

void Registration::getRotateScaleImage(cv::Mat& src, std::vector<cv::Mat>& dst_s, std::vector<std::pair<double, double>> rotate_scale_set)
{
	cv::Point2f src_center(src.cols / 2.0f, src.rows / 2.0f);
	for (auto elem : rotate_scale_set)
	{
		cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, elem.first, elem.second);
		cv::Mat dst;
		cv::warpAffine(src, dst, rot_mat, src.size());
		dst_s.emplace_back(std::move(dst));
	}
}


void Registration::getRotateScaleImage(
	cv::Mat& src, std::vector<cv::Mat>& dst_s, 
	int rotate_num, int scale_num, double angle, double scale, 
	double angle_center, double scale_center)
{
	std::vector<std::pair<double, double>> rotate_scale_set = getRotateScaleSet(rotate_num, scale_num, angle, scale, angle_center, scale_center);
	getRotateScaleImage(src, dst_s, rotate_scale_set);
}

void Registration::getRotateScaleImage(cv::cuda::GpuMat& src, std::vector<cv::cuda::GpuMat>& dst_s, std::vector<std::pair<double, double>> rotate_scale_set)
{
	cv::Point2f src_center(src.cols / 2.0f, src.rows / 2.0f);
	for (auto elem : rotate_scale_set)
	{
		cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, elem.first, elem.second);
		cv::cuda::GpuMat dst;
		cv::cuda::warpAffine(src, dst, rot_mat, src.size());
		dst_s.emplace_back(std::move(dst));
	}
}
void Registration::getRotateScaleImage(
	cv::cuda::GpuMat& src, std::vector<cv::cuda::GpuMat>& dst_s, 
	int rotate_num, int scale_num, double angle, double scale, double angle_center, double scale_center)
{
	std::vector<std::pair<double, double>> rotate_scale_set = getRotateScaleSet(rotate_num, scale_num, angle, scale, angle_center, scale_center);
	getRotateScaleImage(src, dst_s, rotate_scale_set);
}

std::pair<cv::Point, double> Registration::findMatchLocAndValue(cv::cuda::GpuMat& gpu_template_img, cv::cuda::GpuMat& gpu_big_img)
{
	cv::cuda::GpuMat gpu_result;
	int match_method = CV_TM_CCOEFF_NORMED;
	std::unique_lock<mutex> match_lock(match_mutex);
	cv::Ptr<cv::cuda::TemplateMatching> tm_obj = cv::cuda::createTemplateMatching(gpu_template_img.type(), match_method, cv::Size(0, 0));
	tm_obj->match(gpu_big_img, gpu_template_img, gpu_result/*, stream*/);
	match_lock.unlock();
	//if (m_debug_flag)
	//{
	//	cv::Mat template_img;
	//	gpu_template_img.download(template_img);
	//	cv::Mat big_img;
	//	gpu_big_img.download(big_img);		
	//	cv::Mat result;
	//	gpu_result.download(result);
	//	cv::matchTemplate(big_img, template_img, result, match_method);
	//}
	/// Localizing the best match with minMaxLoc
	double minVal = 0.0f; double maxVal = 0.0f; cv::Point minLoc; cv::Point maxLoc;
	cv::Point matchLoc;
	double matchVal = 0.0f;
	cv::cuda::minMaxLoc(gpu_result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
		matchVal = minVal;
	}
	else
	{
		matchLoc = maxLoc;
		matchVal = maxVal;
	}
	std::pair<cv::Point, double> ret;
	ret.first = matchLoc;
	ret.second = matchVal;
	return ret;
}

std::pair<cv::Point, double> Registration::findMatchLocAndValue(cv::Mat& template_img, cv::Mat& big_img)
{
	std::pair<cv::Point, double> ret;
	int match_method = CV_TM_CCOEFF_NORMED;
	cv::Mat result;
	cv::matchTemplate(big_img, template_img, result, match_method);
	double minVal = 0.0f; double maxVal = 0.0f; cv::Point minLoc; cv::Point maxLoc;
	cv::Point matchLoc;
	double matchVal = 0.0f;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
		matchVal = minVal;
	}
	else
	{
		matchLoc = maxLoc;
		matchVal = maxVal;
	}
	ret.first = matchLoc;
	ret.second = matchVal;
	return ret;
}

int Registration::getBiggestLocValue(std::vector<std::pair<cv::Point, double>>& loc_vals)
{
	if (loc_vals.size() == 0)
		return -1;
	int place = 0;
	std::pair<cv::Point, double> final_result;
	for (auto iter = loc_vals.begin(); iter != loc_vals.end(); iter++)
	{
		if (iter->second > final_result.second)
		{
			final_result = *iter;
			place = iter - loc_vals.begin();
		}
	}
	return place;
}

cv::Rect Registration::getCenterRect(int rows, int cols, double angle)
{

	angle = angle / 180.0f * 3.1415926f;
	double cos_theta = cos(angle);
	double sin_theta = sin(angle);
	//下面这些是在坐标系中的x,y
	int x0 = cols / 2;
	int y0 = rows / 2;
	double x_frmt1 = std::pow(x0, 2) / (x0 * cos_theta + y0 * sin_theta);
	double x_frmt2 = x0 * y0 / (y0 * cos_theta + x0 * sin_theta);
	double y_frmt1 = x0 * y0 / (x0 * cos_theta + y0 * sin_theta);
	double y_frmt2 = std::pow(y0, 2) / (y0 * cos_theta + x0 * sin_theta);

	cv::Rect ret_rect(0, 0, 0, 0);
	int dst_img_width = min(x_frmt1, x_frmt2) * 2;
	int dst_img_height = min(y_frmt1, y_frmt2) * 2;

	int start_x = (rows - dst_img_height) / 2;
	int start_y = (cols - dst_img_width) / 2;
	ret_rect.x = start_x;
	ret_rect.y = start_y;
	ret_rect.width = dst_img_width;
	ret_rect.height = dst_img_height;
	return ret_rect;
}

cv::Rect Registration::getRotateImgCenterRect(cv::Mat& img, double angle)
{
	cv::Rect ret_rect = getCenterRect(img.rows, img.cols, angle);
	return ret_rect;
}

cv::Rect Registration::getRotateImgCenterRect(cv::cuda::GpuMat& img, double angle)
{
	cv::Rect ret_rect = getCenterRect(img.rows, img.cols, angle);
	return ret_rect;
}

int Registration::findBestLocVal(
	cv::Mat& adjust_img, cv::Mat& standard_img,
	std::vector<std::pair<double, double>> rotate_scale_set, std::vector<std::pair<cv::Point, double>>& ret_result)
{
	std::vector<cv::Mat> rotate_scale_imgs;
	getRotateScaleImage(adjust_img, rotate_scale_imgs, rotate_scale_set);
	cv::Rect center_rect = getRotateImgCenterRect(adjust_img, max_angle);
	std::vector<cv::Mat> crop_imgs;
	for (auto& elem : rotate_scale_imgs)
	{
		cv::Mat temp_mat = elem(center_rect);
		crop_imgs.emplace_back(temp_mat);
	}
	std::vector<std::pair<cv::Point, double>> loc_vals;
	for (int i = 0; i < crop_imgs.size(); i++)
	{
		std::pair<cv::Point, double> loc_val = findMatchLocAndValue(crop_imgs[i], standard_img);
		loc_vals.emplace_back(loc_val);
	}
	int place = getBiggestLocValue(loc_vals);
	ret_result = loc_vals;
	return place;
}

int Registration::findBestLocVal(
	cv::cuda::GpuMat& adjust_img, cv::cuda::GpuMat& standard_img,
	std::vector<std::pair<double, double>> rotate_scale_set, std::vector<std::pair<cv::Point, double>>& ret_result)
{
	std::vector<cv::cuda::GpuMat> gpu_rotate_scale_imgs;
	getRotateScaleImage(adjust_img, gpu_rotate_scale_imgs, rotate_scale_set);
	cv::Rect center_rect = getRotateImgCenterRect(adjust_img, max_angle);
	std::vector<cv::cuda::GpuMat> gpu_crop_imgs;
	for (auto& elem : gpu_rotate_scale_imgs)
	{
		cv::cuda::GpuMat temp_mat = elem(center_rect);
		gpu_crop_imgs.emplace_back(temp_mat);
	}
	std::vector<std::pair<cv::Point, double>> loc_vals;
	for (int i = 0; i < gpu_crop_imgs.size(); i++)
	{
		std::pair<cv::Point, double> loc_val = findMatchLocAndValue(gpu_crop_imgs[i], standard_img);
		loc_vals.emplace_back(loc_val);
	}
	int place = getBiggestLocValue(loc_vals);
	ret_result = loc_vals;
	return place;
}

bool Registration::matchPhoneBigImage(
	cv::Mat& phone_img, cv::Mat& slide_img, cv::Mat& together_img, std::pair<cv::Point, double>& ret_result)
{
	m_debug_flag = false;
	//第一次匹配
	cv::Mat phone_img_4;//缩小四倍的手机图像
	cv::Mat slide_img_4;//缩小四倍的切片小图
	cv::resize(phone_img, phone_img_4, cv::Size(phone_img.cols / 4, phone_img.rows / 4));
	cv::resize(slide_img, slide_img_4, cv::Size(slide_img.cols / 4, slide_img.rows / 4));
	cv::cuda::GpuMat gpu_phone_img_4(phone_img_4);
	cv::cuda::GpuMat gpu_slide_img_4(slide_img_4);
	std::vector<std::pair<double, double>> rotate_scale_set = getRotateScaleSet(40, 8, 0.05f, 0.01f);
	std::vector<std::pair<cv::Point, double>> loc_vals;
	int place = findBestLocVal(gpu_phone_img_4, gpu_slide_img_4, rotate_scale_set, loc_vals);
	//int place = findBestLocVal(phone_img_4, slide_img_4, rotate_scale_set, loc_vals);
	std::pair<double, double> best_rotate_scale = rotate_scale_set[place];
	//如果第一次匹配到了临界点，那么就应该将其掰回正轨
	if (best_rotate_scale.first == max_angle)
	{
		best_rotate_scale.first = max_angle - 0.05f;
	}
	if (best_rotate_scale.first == max_angle*(-1))
	{
		best_rotate_scale.first = max_angle + 0.05f;
	}
	if (best_rotate_scale.second == max_scale)
	{
		best_rotate_scale.first = max_scale - 0.01f;
	}
	if (best_rotate_scale.second == max_scale *(-1))
	{
		best_rotate_scale.first = max_scale + 0.01f;
	}
	//第二次匹配
	cv::Mat phone_img_2;//缩小二倍的手机图像
	cv::Mat slide_img_2;//缩小二倍的切片小图
	cv::resize(phone_img, phone_img_2, cv::Size(phone_img.cols / 2, phone_img.rows / 2));
	cv::resize(slide_img, slide_img_2, cv::Size(slide_img.cols / 2, slide_img.rows / 2));
	cv::cuda::GpuMat gpu_phone_img_2(phone_img_2);
	cv::cuda::GpuMat gpu_slide_img_2(slide_img_2);
	rotate_scale_set = getRotateScaleSet(10, 4, 0.01f, 0.005f, best_rotate_scale.first, best_rotate_scale.second);
	place = findBestLocVal(gpu_phone_img_2, gpu_slide_img_2, rotate_scale_set, loc_vals);
	//place = findBestLocVal(phone_img_2, slide_img_2, rotate_scale_set, loc_vals);
	best_rotate_scale = rotate_scale_set[place];

	//第三次匹配（原图匹配）
	m_debug_flag = true;
	cv::cuda::GpuMat gpu_phone_img(phone_img);
	cv::cuda::GpuMat gpu_slide_img(slide_img);
	rotate_scale_set = getRotateScaleSet(4, 4, 0.005f, 0.0025f, best_rotate_scale.first, best_rotate_scale.second);
	//place = findBestLocVal(gpu_phone_img, gpu_slide_img, rotate_scale_set, loc_vals);
	place = findBestLocVal(phone_img, slide_img, rotate_scale_set, loc_vals);
	ret_result = loc_vals[place];
	//if (ret_result.second < m_threshold_2)
	//	return false;
	//保存图像
	cv::Point2f src_center(gpu_phone_img.cols / 2.0f, gpu_phone_img.rows / 2.0f);
	cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, rotate_scale_set[place].first, rotate_scale_set[place].second);
	cv::Mat dst;
	cv::warpAffine(phone_img, dst, rot_mat, phone_img.size());
	cv::Rect center_rect = getRotateImgCenterRect(phone_img, max_angle);
	dst = dst(center_rect).clone();

	cv::Mat final_img;
	final_img.push_back(dst);
	final_img.push_back(slide_img(cv::Rect(loc_vals[place].first.x, loc_vals[place].first.y, center_rect.width, center_rect.height)));
	together_img = std::move(final_img);
	return true;
}

bool Registration::findFinalImg(cv::Mat& small_img, cv::Mat& big_img, cv::Mat &together_img, std::pair<cv::Point, double>& ret_result)
{
	vector<cv::Mat> rotate_scale_imgs;
	getRotateScaleImage(small_img, rotate_scale_imgs, 40, 8, 0.05f, 0.01f, 0, 0);
	//vector<cv::cuda::GpuMat> gpu_rotate_scale_imgs;
	//cv::cuda::GpuMat gpu_small_img(small_img);
	//getRotateScaleImage(gpu_small_img, gpu_rotate_scale_imgs, 40, 8, 0.05f, 0.01f);
	
	cv::cuda::GpuMat gpu_big_img(big_img);
	////对每个rotate_scale_imgs进行裁剪512*512(中心裁剪)
	int dst_img_size = 512;
	int start_x = (small_img.rows - dst_img_size) / 2;
	int start_y = (small_img.cols - dst_img_size) / 2;
	vector<cv::Mat> crop_imgs;

	for (auto& elem : rotate_scale_imgs)
	{
		cv::Mat temp_mat = elem(cv::Rect(start_x, start_y, 512, 512));//先不尝试clone深拷贝
		crop_imgs.emplace_back(temp_mat);
	}	
	//vector<cv::cuda::GpuMat> gpu_crop_imgs;
	//for (auto& elem : gpu_rotate_scale_imgs)
	//{
	//	cv::cuda::GpuMat temp_mat = elem(cv::Rect(start_x, start_y, dst_img_size, dst_img_size));//先不尝试clone深拷贝
	//	gpu_crop_imgs.emplace_back(temp_mat);
	//}

	vector<std::pair<cv::Point, double>> loc_vals;

	for (int i = 0; i < crop_imgs.size(); i++)
	{
		//cv::cuda::GpuMat gpu_template_img(crop_imgs[i]);
		std::pair<cv::Point, double> loc_val = findMatchLocAndValue(crop_imgs[i], big_img);
		loc_vals.emplace_back(loc_val);
	}
	//将拥有最大分数的返回回去
	assert(loc_vals == 0);
	//std::pair<cv::Point, double> final_result = loc_vals[0];
	int place = getBiggestLocValue(loc_vals);
	ret_result = loc_vals[place];
	if (ret_result.second < m_threshold_2)
		return false;
	cv::Mat final_mat;
	//cv::Mat final_gpu_mat;
	//crop_imgs[place].download(final_gpu_mat);
	final_mat.push_back(crop_imgs[place]);
	final_mat.push_back(big_img(cv::Rect(ret_result.first.x, ret_result.first.y, crop_imgs[place].cols, crop_imgs[place].rows)));
	together_img = std::move(final_mat);
	return true;
}

void Registration::saveImg(string savePath)
{

}

std::pair<cv::Point, double> Registration::adjustPhoneImgOri(cv::Mat& phone_img)
{
	float phone_slide_ratio = m_phone_img_mpp / m_slide_mpp;
	float scale_ratio = std::pow(m_read_level, m_slide_ratio);
	int phone_img_new_height = phone_img.rows * phone_slide_ratio / scale_ratio;
	int phone_img_new_width = phone_img.cols * phone_slide_ratio / scale_ratio;
	cv::Mat new_phone_img;
	cv::resize(phone_img, new_phone_img, cv::Size(phone_img_new_width, phone_img_new_height));
	//还要搞几个旋转的图像
	vector<cv::Mat> rotate_imgs;
	enum RotateFlags {
		ROTATE_90_CLOCKWISE = 0, //Rotate 90 degrees clockwise
		ROTATE_180 = 1, //Rotate 180 degrees clockwise
		ROTATE_90_COUNTERCLOCKWISE = 2, //Rotate 270 degrees clockwise
	};
	rotate_imgs.emplace_back(new_phone_img);
	for (int i = 0; i <= 2; i++)
	{
		cv::Mat temp_mat;
		cv::rotate(new_phone_img, temp_mat, i);
		rotate_imgs.emplace_back(std::move(temp_mat));
	}
	std::pair<cv::Point, double> loc_val(std::make_pair(cv::Point(0, 0), 0.0f));
	int place = 0;
	for (auto iter = rotate_imgs.begin(); iter != rotate_imgs.end(); iter++)
	{
		cv::cuda::GpuMat gpu_template_img(*iter);
		std::pair<cv::Point, double> temp_loc_val = findMatchLocAndValue(gpu_template_img, m_gpu_level_img);
		if (temp_loc_val.second > loc_val.second)
		{
			loc_val = temp_loc_val;
			place = iter - rotate_imgs.begin();
		}
	}
	if (place != 0)
	{
		cv::rotate(phone_img, phone_img, place - 1);
	}
	return loc_val;
}

void Registration::getSlideTile(int read_handle_index, int read_level, cv::Rect rect, cv::Mat& read_img)
{
	std::unique_lock<std::mutex> read_lock(read_mutexs[read_handle_index]);
	read_handles[read_handle_index]->getTile(read_level, rect.x, rect.y, rect.width, rect.height, read_img);
	read_lock.unlock();
}

void Registration::popData(cv::Mat& img)
{
	std::unique_lock<mutex> data_lock(data_mutex);
	img = std::move(data_queue.front());
	data_queue.pop();
	data_lock.unlock();
}

std::pair<float, float> Registration::getRatios()
{
	float phone_slide_ratio = m_phone_img_mpp / m_slide_mpp;
	float scale_ratio = std::pow(m_read_level, m_slide_ratio);
	std::pair<float, float> ret_ratio(std::make_pair(phone_slide_ratio, scale_ratio));
	return ret_ratio;
}

cv::Point Registration::getMatchedSlideImage(cv::Mat& phone_img, cv::Mat& slide_img, int read_handle_index)
{
	auto ret_ratio = getRatios();
	popData(phone_img);
	std::pair<cv::Point, double> loc_val = adjustPhoneImgOri(phone_img);
	//3.从切片图像中扣取找到的位置的图像
	int find_img_x = ret_ratio.second * loc_val.first.x;
	int find_img_y = ret_ratio.second * loc_val.first.y;
	int find_img_height = phone_img.rows * ret_ratio.first;
	int find_img_width = phone_img.cols * ret_ratio.first;
	getSlideTile(0, 0, cv::Rect(find_img_x, find_img_y, find_img_width, find_img_height), slide_img);
	cv::resize(slide_img, slide_img, cv::Size(phone_img.cols, phone_img.rows));
	return cv::Point(find_img_x, find_img_y);
}

void Registration::process_single_data2(int read_handle_index)
{
	cv::Mat phone_img;
	cv::Mat slide_img;
	cv::Mat together_img;
	auto ratios = getRatios();
	cv::Point point = getMatchedSlideImage(phone_img, slide_img, read_handle_index);
	std::pair<cv::Point, double> result;
	bool flag = matchPhoneBigImage(phone_img, slide_img, together_img, result);
	if (flag)
	{
		cout << m_write_num << " ";
		int save_img_x = point.x + result.first.x * ratios.first;
		int save_img_y = point.y + result.first.y * ratios.first;
		std::string save_img_name = to_string(save_img_x) + "_" + to_string(save_img_y) + "_" + to_string(result.second) + ".tif";
		cv::imwrite(m_dst_path + "\\" + save_img_name, together_img);
		m_write_num++;
	}
}

void Registration::process_single_data(int read_handle_index)
{
	//2.将手机图像和这个l4图像做matchTemplate
	cv::Mat phone_img;
	cv::Mat slide_img;
	cv::Point point = getMatchedSlideImage(phone_img, slide_img, read_handle_index);
	//4.按照一定规则，裁取手机图像600*600，然后从find_img中寻找与其对应的512*512图像
	int big_rect_size = 600;
	vector<cv::Rect> rects = iniRects(big_rect_size, big_rect_size, phone_img.rows, phone_img.cols, big_rect_size * 0.25f);
	vector<cv::Mat> phone_crop_imgs;
	vector<cv::Mat> find_crop_imgs;
	for (auto elem : rects)
	{
		phone_crop_imgs.emplace_back(phone_img(elem));
		find_crop_imgs.emplace_back(slide_img(elem));
	}
	vector<std::pair<cv::Point, double>> results;//最终的结果
	cv::Mat together_img;
	//vector<cv::Mat> together_imgs;
	for (int i = 0; i < phone_crop_imgs.size(); i++)
	{
		std::pair<cv::Point, double> ret_result;
		bool flag = findFinalImg(phone_crop_imgs[i], find_crop_imgs[i], together_img, ret_result);
		if (flag)
		{
			cout << m_write_num << " ";
			int save_img_x = point.x + rects[i].x + ret_result.first.x;
			int save_img_y = point.y + rects[i].y + ret_result.first.y;
			std::string save_img_name = to_string(save_img_x) + "_" + to_string(save_img_y) + ".tif";
			//std::string save_img_path = "D:\\TEST_OUTPUT\\MJB_Registration\\";
			cv::imwrite(m_dst_path + "\\" + save_img_name, together_img);
			m_write_num++;
		}
	}
	//cout << endl;
}

void Registration::initReadHandle()
{
	std::unique_ptr<SlideFactory> sFactory(new SlideFactory());
	if (read_handles.size() == 0)
	{
		for (int i = 0; i < totalThrNum; i++)
		{
			read_handles.emplace_back(sFactory->createSlideProduct(m_slide_path.c_str()));
		}
		std::vector<std::mutex> list(read_handles.size());
		read_mutexs.swap(list);
	}
	else
	{
		for (int i = 0; i < totalThrNum; i++)
		{
			read_handles[i].reset();
			read_handles[i] = sFactory->createSlideProduct(m_slide_path.c_str());
		}
	}
}

void Registration::initL4Img()
{
	//1.读取指定level的图像
	int l4_width = 0;
	int l4_height = 0;
	read_handles[0]->getLevelDimensions(m_read_level, l4_width, l4_height);
	std::unique_lock<mutex> read_lock(read_mutexs[0]);
	read_handles[0]->getTile(m_read_level, 0, 0, l4_width, l4_height, m_level_img);
	m_gpu_level_img.upload(m_level_img);
	read_lock.unlock();
}

void Registration::process(std::string phone_img_path, std::string sdpc_path, std::string dst_path)
{
	//
	m_slide_path = sdpc_path;
	m_dst_path = dst_path;
	createDirRecursive(m_dst_path);
	int define_thread_num = 2;
	std::call_once(create_thread_flag, &Registration::createThreadPool, this, define_thread_num);
	initReadHandle();
	initL4Img();
	m_slide_ratio = read_handles[0]->m_ratio;
	read_handles[0]->getSlideMpp(m_slide_mpp);

	GetPhoneData phone_data;
	phone_data.process(phone_img_path);
	std::vector<cv::Mat> imgs;
	while (phone_data.popData(imgs))
	{
		int size = imgs.size();
		//cout << "imgs size:" << size << endl;
		std::unique_lock<mutex> data_lock(data_mutex);
		for (int i = 0; i < size; i++)
		{
			data_queue.emplace(std::move(imgs[i]));
		}
		data_lock.unlock();
		std::unique_lock<mutex> task_lock(task_mutex);
		for (int i = 0; i < size; i++)
		{		
			auto task = std::make_shared<std::packaged_task<void()>>(std::bind(&Registration::process_single_data2, this, i % define_thread_num));
			tasks.emplace(
				[task]() {
					(*task)();
				}
			);
		}
		task_lock.unlock();
		task_cv.notify_all();
		imgs.clear();
	}
	while (idlThrNum != totalThrNum)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	}
}

vector<cv::Rect> Registration::iniRects(int sHeight, int sWidth, int height, int width, int overlap, bool flag_right, bool flag_down)
{
	vector<cv::Rect> rects;
	//进行参数检查
	if (sHeight == 0 || sWidth == 0 || height == 0 || width == 0)
	{
		cout << "iniRects: parameter should not be zero\n";
		return rects;
	}
	if (sHeight > height || sWidth > width)
	{
		cout << "iniRects: sHeight or sWidth > height or width\n";
		return rects;
	}
	int x_num = (width - overlap) / (sWidth - overlap);
	int y_num = (height - overlap) / (sHeight - overlap);
	vector<int> xStart;
	vector<int> yStart;
	if ((x_num * (sWidth - overlap) + overlap) == width)
	{
		flag_right = false;
	}
	if ((y_num * (sHeight - overlap) + overlap) == height)
	{
		flag_down = false;
	}
	for (int i = 0; i < x_num; i++)
	{
		xStart.emplace_back((sWidth - overlap) * i);
	}
	for (int i = 0; i < y_num; i++)
	{
		yStart.emplace_back((sHeight - overlap) * i);
	}
	if (flag_right)
		xStart.emplace_back(width - sWidth);
	if (flag_down)
		yStart.emplace_back(height - sHeight);
	for (int i = 0; i < yStart.size(); i++)
	{
		for (int j = 0; j < xStart.size(); j++)
		{
			cv::Rect rect;
			rect.x = xStart[j];
			rect.y = yStart[i];
			rect.width = sWidth;
			rect.height = sHeight;
			rects.emplace_back(rect);
		}
	}
	return rects;
}

