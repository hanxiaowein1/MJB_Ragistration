#include "TaskThread.h"


TaskThread::~TaskThread()
{
	stopped.store(true);
	task_cv.notify_all();
	for (std::thread& thread : pool) {
		if (thread.joinable())
			thread.join();

	}
}

void TaskThread::createThreadPool(int thread_num)
{
	if (my_once_flag == true)
	{
		return;
	}
	my_once_flag = true;
	//在内部确定其只能被调用一次
	idlThrNum = thread_num;
	totalThrNum = thread_num;
	stopped.store(false);
	for (int size = 0; size < totalThrNum; ++size)
	{   //初始化线程数量
		pool.emplace_back(
			[this]
			{ // 工作线程函数
				while (!this->stopped.load())
				{
					std::function<void()> task;
					{   // 获取一个待执行的 task
						std::unique_lock<std::mutex> lock{ this->task_mutex };// unique_lock 相比 lock_guard 的好处是：可以随时 unlock() 和 lock()
						this->task_cv.wait(lock,
							[this] {
								return this->stopped.load() || !this->tasks.empty();
							}
						); // wait 直到有 task
						if (this->stopped.load() && this->tasks.empty())
							return;
						task = std::move(this->tasks.front()); // 取一个 task
						this->tasks.pop();
					}
					idlThrNum--;
					task();
					idlThrNum++;
				}
			}
			);
	}
}