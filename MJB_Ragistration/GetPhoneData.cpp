#include "GetPhoneData.h"
#include "commonFunction.h"

using namespace std;
void GetPhoneData::readImg(std::string img_path)
{
	cv::Mat read_img = cv::imread(img_path);
	pushData(read_img);
}

void GetPhoneData::process(std::string imgs_path, std::string suffix)
{
	std::call_once(create_thread_flag, &GetPhoneData::createThreadPool, this, 4);
	std::vector<std::string> img_paths;
	getFiles(imgs_path, img_paths, suffix);
	std::unique_lock<mutex> task_lock(task_mutex);
	for (int i = 0; i < img_paths.size(); i++)
	{
		auto task = std::make_shared<std::packaged_task<void()>>(std::bind(&GetPhoneData::readImg, this, img_paths[i]));
		tasks.emplace(
			[task]() {
				(*task)();
			}
		);
	}
	task_lock.unlock();
	task_cv.notify_all();
}