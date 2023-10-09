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

#include "config/logging.h"

#include <catch2/catch_session.h>

int main(int argc, char* argv[]) {
    Catch::Session session;

    bool enable_logging(false);

    auto cli = session.cli()
               | Catch::Clara::Opt(enable_logging)["--enable-logging"]("Enable logging output during testing");

    session.cli(cli);

    auto returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) {
        return returnCode;
    }

#ifdef ENABLE_LOGGING
    if (enable_logging) {
        spdlog::default_logger()->set_level(spdlog::level::trace);
    } else {
        spdlog::default_logger()->set_level(spdlog::level::off);
    }
#endif  // ENABLE_LOGGING

    return session.run();
}
