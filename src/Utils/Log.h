// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <exception>
#include <fstream>
#include <mutex>

#include "tinyformat/tinyformat.h"

#ifdef ERROR
#undef ERROR
#endif

namespace Varjo
{
	enum class LogMessageLevel { DEBUG, INFO, WARNING, ERROR };

	class Log
	{
	public:

		Log();
		explicit Log(const std::string& logFileName);

		void setLogFile(const std::string& logFileName);
		void setMinimumMessageLevel(LogMessageLevel value);

		void logMessage(LogMessageLevel messageLevel, const std::string& message);
		void logDebug(const std::string& message);
		void logInfo(const std::string& message);
		void logWarning(const std::string& message);
		void logError(const std::string& message);
		void logException(const std::exception_ptr& exception);

		template <typename... Args>
		void logMessage(LogMessageLevel messageLevel, const std::string& message, const Args&... args);

		template <typename... Args>
		void logDebug(const std::string& message, const Args&... args);

		template <typename... Args>
		void logInfo(const std::string& message, const Args&... args);

		template <typename... Args>
		void logWarning(const std::string& message, const Args&... args);

		template <typename... Args>
		void logError(const std::string& message, const Args&... args);

	private:

		void handleMessage(LogMessageLevel messageLevel, const std::string& message);
		std::string formatMessage(LogMessageLevel messageLevel, const std::string& message);
		void outputMessage(LogMessageLevel messageLevel, const std::string& message);

		std::ofstream logFile;
		std::mutex outputMutex;
		LogMessageLevel minimumMessageLevel = LogMessageLevel::DEBUG;
	};
}

#include "Utils/Log.inl"
