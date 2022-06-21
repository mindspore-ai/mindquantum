//   Copyright 2021 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#ifndef CLANG_VERSION_HPP
#define CLANG_VERSION_HPP

#ifdef __clang__
#    ifdef __apple_build_version__
#        define mq_clang_version (10 * (10 * (__clang_major__) + (__clang_minor__)) + (__clang_patchlevel__))
#        if mq_clang_version < 700
#            error Version of clang is too old!
#        elif mq_clang_version < 900
#            define MQ_CLANG_MAJOR 3
#        elif mq_clang_version == 900
#            define MQ_CLANG_MAJOR 4
#        elif mq_clang_version < 1000
#            define MQ_CLANG_MAJOR 5
#        elif mq_clang_version == 1000
#            define MQ_CLANG_MAJOR 6
#        elif mq_clang_version == 1001
#            define MQ_CLANG_MAJOR 7
#        elif mq_clang_version == 1100
#            define MQ_CLANG_MAJOR 8
#        elif mq_clang_version == 1103
#            define MQ_CLANG_MAJOR 9
#        elif mq_clang_version == 1200
#            define MQ_CLANG_MAJOR 10
#        elif mq_clang_version == 1205
#            define MQ_CLANG_MAJOR 11
#        elif mq_clang_version == 1300
#            define MQ_CLANG_MAJOR 12
#        elif mq_clang_version == 1316
#            define MQ_CLANG_MAJOR 13
#        else
#            define MQ_CLANG_MAJOR 13

#        endif
#        if mq_clang_version < 702
#            define MQ_CLANG_MINOR 7
#        elif mq_clang_version == 730
#            define MQ_CLANG_MINOR 8
#        elif mq_clang_version <= 810
#            define MQ_CLANG_MINOR 9
#        elif mq_clang_version <= 1200
#            define MQ_CLANG_MINOR 0
#        elif mq_clang_version == 1205
#            define MQ_CLANG_MINOR 1
#        else
#            define MQ_CLANG_MINOR 0
#        endif
#        undef mq_clang_version
#    else
#        define MQ_CLANG_MAJOR __clang_major__
#        define MQ_CLANG_MINOR __clang_minor__
#    endif  // __apple_build_version__
#endif      // __clang__

#endif /* CLANG_VERSION_HPP */
