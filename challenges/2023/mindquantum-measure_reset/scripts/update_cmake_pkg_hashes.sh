#! /bin/bash

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )

cmake "$@" -P "${BASEPATH}/cmake/update_cmake_pkg_hashes.cmake"
