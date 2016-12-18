// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

#ifdef __linux
#include <unistd.h>
#endif

#include <sys/stat.h>
#include <boost/filesystem.hpp>

#include "Utils/SysUtils.h"
#include "Utils/Log.h"
#include "Utils/App.h"

using namespace Varjo;
namespace bf = boost::filesystem;

void SysUtils::openFileExternally(const std::string& fileName)
{
	Log& log = App::getLog();
	log.logInfo("Opening file in an external viewer (%s)", fileName);

#ifdef _WIN32
	ShellExecuteA(nullptr, "open", fileName.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
	int32_t pid = fork();

	if (pid == 0)
	{
#ifdef __linux
		char* arg[] = { (char*)"xdg-open", (char*)fileName.c_str(), (char*)nullptr };
#elif __APPLE__
        char* arg[] = { (char*)"open", (char*)fileName.c_str(), (char*)nullptr };
#endif
		if (execvp(arg[0], arg) == -1)
			log.logWarning("Could not open file externally (%d)", errno);
	}
#endif
}

void SysUtils::setConsoleTextColor(ConsoleTextColor color)
{
	(void)color;

#ifdef _WIN32
	HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);

	if (consoleHandle == nullptr)
		return;

	switch (color)
	{
		case ConsoleTextColor::DEFAULT:
			SetConsoleTextAttribute(consoleHandle, 7 + 0 * 16);
			break;

		case ConsoleTextColor::GRAY_ON_BLACK:
			SetConsoleTextAttribute(consoleHandle, 8 + 0 * 16);
			break;

		case ConsoleTextColor::WHITE_ON_BLACK:
			SetConsoleTextAttribute(consoleHandle, 15 + 0 * 16);
			break;

		case ConsoleTextColor::YELLOW_ON_BLACK:
			SetConsoleTextAttribute(consoleHandle, 14 + 0 * 16);
			break;

		case ConsoleTextColor::WHITE_ON_RED:
			SetConsoleTextAttribute(consoleHandle, 15 + 12 * 16);
			break;

		default: break;
	}
#endif
}

uint64_t SysUtils::getFileSize(const std::string& fileName)
{
	struct stat stat_buf;
	int rc = stat(fileName.c_str(), &stat_buf);
	return rc == 0 ? stat_buf.st_size : 0;
}

std::vector<std::string> SysUtils::getAllFiles(const std::string& dirName)
{
	std::vector<std::string> files;
	bf::path directory(dirName);
	bf::directory_iterator end;

	for (bf::directory_iterator it(directory); it != end; ++it)
	{
		if (bf::is_regular_file(it->path()))
			files.push_back(bf::absolute(it->path()).string());
	}

	return files;
}
