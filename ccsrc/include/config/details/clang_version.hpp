/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef MQ_CONFIG_CLANG_VERSION_HPP
#define MQ_CONFIG_CLANG_VERSION_HPP

#ifdef __clang__
#    ifdef __apple_build_version__
#        if __apple_build_version__ > 14000000
#            define MQ_CLANG_MAJOR 14
#        elif __apple_build_version__ > 13160000
#            define MQ_CLANG_MAJOR 13
#        elif __apple_build_version__ > 13000000
#            define MQ_CLANG_MAJOR 12
#        elif __apple_build_version__ > 12050000
#            define MQ_CLANG_MAJOR 11
#        elif __apple_build_version__ > 12000000
#            define MQ_CLANG_MAJOR 10
#        elif __apple_build_version__ > 11030000
#            define MQ_CLANG_MAJOR 9
#        elif __apple_build_version__ > 11000000
#            define MQ_CLANG_MAJOR 8
#        elif __apple_build_version__ > 10010000
#            define MQ_CLANG_MAJOR 7
#        else
#            error Detected version of clang is too old!
#        endif
#        if (__apple_build_version__ / 10000) == 1205
#            define MQ_CLANG_MINOR 1
#        else
#            define MQ_CLANG_MINOR 0
#        endif
#    else
#        define MQ_CLANG_MAJOR __clang_major__
#        define MQ_CLANG_MINOR __clang_minor__
#    endif  // __apple_build_version__
#endif      // __clang__

#endif /* MQ_CONFIG_CLANG_VERSION_HPP */
