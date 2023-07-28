/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MQ_LOGGING_HPP
#define MQ_LOGGING_HPP

#ifdef ENABLE_LOGGING
#    include <string>

// NB: setting this to WARN will disable at compile time any log calls lower than that
#    if !(defined MQ_LOG_ACTIVE_LEVEL) && (defined SPDLOG_ACTIVE_LEVEL)
#        define MQ_LOG_ACTIVE_LEVEL SPDLOG_ACTIVE_LEVEL
#    endif  // !MQ_LOG_ACTIVE_LEVEL && SPDLOG_ACTIVE_LEVEL
#    if !(defined SPDLOG_ACTIVE_LEVEL) && (defined MQ_LOG_ACTIVE_LEVEL)
#        define SPDLOG_ACTIVE_LEVEL MQ_LOG_ACTIVE_LEVEL
#    endif  // !SPDLOG_ACTIVE_LEVEL && MQ_LOG_ACTIVE_LEVEL

#    if !(defined SPDLOG_ACTIVE_LEVEL) && !(defined MQ_LOG_ACTIVE_LEVEL)
#        define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN
#        define MQ_LOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN
#    endif  // !SPDLOG_ACTIVE_LEVEL && !MQ_LOG_ACTIVE_LEVEL

#    include <spdlog/spdlog.h>

namespace mindquantum::logging {
void set_log_file(const std::string& filename);
}  // namespace mindquantum::logging

#    define MQ_TRACE(...)                SPDLOG_TRACE(__VA_ARGS__)
#    define MQ_LOGGER_TRACE(logger, ...) SPDLOG_LOGGER_TRACE(logger, __VA_ARGS__)
#    define MQ_DEBUG(...)                SPDLOG_DEBUG(__VA_ARGS__)
#    define MQ_LOGGER_DEBUG(logger, ...) SPDLOG_LOGGER_DEBUG(logger, __VA_ARGS__)
#    define MQ_ERROR(...)                SPDLOG_ERROR(__VA_ARGS__)
#    define MQ_LOGGER_ERROR(logger, ...) SPDLOG_LOGGER_ERROR(logger, __VA_ARGS__)
#    define MQ_INFO(...)                 SPDLOG_INFO(__VA_ARGS__)
#    define MQ_LOGGER_INFO(logger, ...)  SPDLOG_LOGGER_INFO(logger, __VA_ARGS__)
#    define MQ_WARN(...)                 SPDLOG_WARN(__VA_ARGS__)
#    define MQ_LOGGER_WARN(logger, ...)  SPDLOG_LOGGER_WARN(logger, __VA_ARGS__)
#else
#    define MQ_TRACE(...)                static_cast<void>(0)
#    define MQ_LOGGER_TRACE(logger, ...) static_cast<void>(0)
#    define MQ_DEBUG(...)                static_cast<void>(0)
#    define MQ_LOGGER_DEBUG(logger, ...) static_cast<void>(0)
#    define MQ_ERROR(...)                static_cast<void>(0)
#    define MQ_LOGGER_ERROR(logger, ...) static_cast<void>(0)
#    define MQ_INFO(...)                 static_cast<void>(0)
#    define MQ_LOGGER_INFO(logger, ...)  static_cast<void>(0)
#    define MQ_WARN(...)                 static_cast<void>(0)
#    define MQ_LOGGER_WARN(logger, ...)  static_cast<void>(0)
#endif  // ENABLE_LOGGING

#endif /* MQ_LOGGING_HPP */
