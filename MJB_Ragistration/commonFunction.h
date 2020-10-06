#ifndef _COMMONFUNC_H_
#define _COMMONFUNC_H_


#include <string>
#include <iostream>
#include <vector>
#include <io.h>
#include <windows.h>

void saveAsTxt(std::vector<float> &score2, std::string name);
void splitString(std::string str, std::string c, std::vector<std::string> *outStr);
std::string getFileNamePrefix(std::string *path);
void getFiles(std::string path, std::vector<std::string> &files, std::string suffix);
void createDirRecursive(std::string dir);
std::string getFileName(std::string path);
void filterList(std::vector<std::string> &solveList, std::vector<std::string> &solvedList);
std::string getFileNameSuffix(std::string path);
#endif