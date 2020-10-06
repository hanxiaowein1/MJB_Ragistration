#pragma once
#ifndef _AILAB_TASKTHREAD_H_
#define _AILAB_TASKTHREAD_H_

#include <vector>
#include <mutex>
#include <atomic>
#include <future>
#include <queue>

class TaskThread
{
public:
	//alias
	using Task = std::function<void()>;
	//thread pool
	std::vector<std::thread> pool;
	// task
	std::condition_variable task_cv;
	std::queue<Task> tasks;
	std::mutex task_mutex;
	std::atomic<bool> stopped;//停止线程的标志
	std::atomic<int> idlThrNum = 1;//闲置线程数量
	std::atomic<int> totalThrNum = 1;//总共线程数量
	std::once_flag create_thread_flag;
	bool my_once_flag = false;
public:
	void createThreadPool(int threadNum);
	virtual ~TaskThread();
};

#endif