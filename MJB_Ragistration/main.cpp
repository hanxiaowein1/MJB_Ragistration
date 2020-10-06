#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "Registration.h"

int main(int args, char* argv[])
{
	if (args == 1)
	{
		_putenv_s("CUDA_VISIBLE_DEVICES", "0");
		std::string sdpc_path = "D:\\TEST_DATA\\sdpc\\1909060011.sdpc";
		std::string phone_img_path = "D:\\TEST_DATA\\MJB_Registration\\1909060011\\";
		std::string dst_path = "D:\\TEST_OUTPUT\\MJB_Registration\\";
		Registration regis;
		regis.process(phone_img_path, sdpc_path, dst_path);
		system("pause");
		return 0;
	}
	else if (args == 4)
	{
		std::string phone_img_path = argv[1];
		std::string sdpc_path = argv[2];
		std::string dst_path = argv[3];
		Registration regis;
		regis.process(phone_img_path, sdpc_path, dst_path);
		return 0;
	}
	else
	{
		std::cout << "usage: executable phoneImagePath singleSdpcPath dstPath\n";
		return -1;
	}
}

int main_old()
{

	system("pause");
	return 0;
}