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
	//���ڲ�ȷ����ֻ�ܱ�����һ��
	idlThrNum = thread_num;
	totalThrNum = thread_num;
	stopped.store(false);
	for (int size = 0; size < totalThrNum; ++size)
	{   //��ʼ���߳�����
		pool.emplace_back(
			[this]
			{ // �����̺߳���
				while (!this->stopped.load())
				{
					std::function<void()> task;
					{   // ��ȡһ����ִ�е� task
						std::unique_lock<std::mutex> lock{ this->task_mutex };// unique_lock ��� lock_guard �ĺô��ǣ�������ʱ unlock() �� lock()
						this->task_cv.wait(lock,
							[this] {
								return this->stopped.load() || !this->tasks.empty();
							}
						); // wait ֱ���� task
						if (this->stopped.load() && this->tasks.empty())
							return;
						task = std::move(this->tasks.front()); // ȡһ�� task
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