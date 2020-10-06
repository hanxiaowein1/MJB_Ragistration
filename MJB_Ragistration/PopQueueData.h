#pragma once

#ifndef _AILAB_POPQUEUEDATA_H_
#define _AILAB_POPQUEUEDATA_H_

#include "TaskThread.h"

template <typename T>
class PopQueueData : public TaskThread
{
public:
	std::queue<T> data_queue;
	std::mutex data_mutex;
	std::condition_variable data_cv;
public:
	void popQueueWithoutLock(std::vector<T>& pop_datas);
	void popQueueWithoutLock(T& pop_data);
	bool popData(std::vector<T>& pop_datas);
	bool popData(T& pop_data);
	void pushData(std::vector<T>& push_datas);
	void pushData(T& push_data);
};


template <typename T>
void PopQueueData<T>::popQueueWithoutLock(T& pop_data)
{
	if (!data_queue.empty())
	{
		pop_data = std::move(data_queue.front());
		data_queue.pop();
	}
}

template <typename T>
void PopQueueData<T>::popQueueWithoutLock(std::vector<T>& pop_datas)
{
	int size = data_queue.size();
	for (int i = 0; i < size; i++)
	{
		pop_datas.emplace_back(std::move(data_queue.front()));
		data_queue.pop();
	}
}

template <typename T>
void PopQueueData<T>::pushData(std::vector<T>& push_datas)
{
	std::unique_lock<std::mutex> data_lock(data_mutex);
	for (auto iter = push_datas.begin(); iter != push_datas.end(); iter++)
	{
		data_queue.emplace(std::move(*iter));
	}
	data_lock.unlock();
	data_cv.notify_one();
}

template <typename T>
void PopQueueData<T>::pushData(T& push_data)
{
	std::unique_lock<std::mutex> data_lock(data_mutex);
	data_queue.emplace(std::move(push_data));
	data_lock.unlock();
	data_cv.notify_one();
}

template <typename T>
bool PopQueueData<T>::popData(T& pop_data)
{
	std::unique_lock<std::mutex> data_lock(data_mutex);
	if (data_queue.size() > 0)
	{
		popQueueWithoutLock(pop_data);
		data_lock.unlock();
		return true;
	}
	else
	{
		data_lock.unlock();
		//取得tasks的锁，检查是否还有任务
		std::unique_lock<std::mutex> task_lock(task_mutex);
		if (tasks.size() > 0)
		{
			task_lock.unlock();
			//证明有task，那么data_mutex再次锁上，因为一定会有数据唤醒它
			data_lock.lock();
			data_cv.wait(data_lock, [this] {
				if (data_queue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
				});
			if (stopped.load())
				return false;
			popQueueWithoutLock(pop_data);
			data_lock.unlock();
			return true;
		}
		else
		{
			task_lock.unlock();
			//如果没有task，那么要检查是否有线程在运行
			if (idlThrNum.load() == totalThrNum.load())
			{
				//如果没有线程在运行，那么在看看队列是否有元素（万一在判断的时候进入了队列了呢？）
				data_lock.lock();//其实不用加锁，因为没有线程在运行，肯定不会占用锁了
				if (data_queue.size() > 0)
				{
					popQueueWithoutLock(pop_data);
					data_lock.unlock();
					return true;
				}
				return false;
			}
			else
			{
				//如果有线程在运行，那么在锁住队列，等待人来唤醒
				data_lock.lock();
				data_cv.wait_for(data_lock, 1000ms, [this] {
					if (data_queue.size() > 0 || stopped.load()) {
						return true;
					}
					else {
						return false;
					}
					});
				if (stopped.load())
					return false;
				data_lock.unlock();
				return popData(pop_data);
			}
		}
	}
}

template <typename T>
bool PopQueueData<T>::popData(std::vector<T>& pop_datas)
{
	std::unique_lock<std::mutex> data_lock(data_mutex);
	if (data_queue.size() > 0)
	{
		popQueueWithoutLock(pop_datas);
		data_lock.unlock();
		return true;
	}
	else
	{
		data_lock.unlock();
		//取得tasks的锁，检查是否还有任务
		std::unique_lock<std::mutex> task_lock(task_mutex);
		if (tasks.size() > 0)
		{
			task_lock.unlock();
			//证明有task，那么data_mutex再次锁上，因为一定会有数据唤醒它
			data_lock.lock();
			data_cv.wait(data_lock, [this] {
				if (data_queue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
				});
			if (stopped.load())
				return false;
			popQueueWithoutLock(pop_datas);
			data_lock.unlock();
			return true;
		}
		else
		{
			task_lock.unlock();
			//如果没有task，那么要检查是否有线程在运行
			if (idlThrNum.load() == totalThrNum.load())
			{
				//如果没有线程在运行，那么在看看队列是否有元素（万一在判断的时候进入了队列了呢？）
				data_lock.lock();//其实不用加锁，因为没有线程在运行，肯定不会占用锁了
				popQueueWithoutLock(pop_datas);
				data_lock.unlock();
				if (pop_datas.size() == 0)
					return false;
				return true;
			}
			else
			{
				//如果有线程在运行，那么在锁住队列，等待人来唤醒
				data_lock.lock();
				data_cv.wait_for(data_lock, 1000ms, [this] {
					if (data_queue.size() > 0 || stopped.load()) {
						return true;
					}
					else {
						return false;
					}
					});
				if (stopped.load())
					return false;
				//popQueueWithoutLock(rectMats);
				data_lock.unlock();
				return popData(pop_datas);
				//return true;
			}
		}
	}
}


#endif