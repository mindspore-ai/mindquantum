cmake_minimum_required(VERSION 3.18)
project(HUBO_Solver LANGUAGES CXX CUDA)

# Find Python and pybind11
# FIX 1: Add this line to use modern Python finding (suppresses CMP0148)
set(pybind11_FINDPYTHON ON)

# You may need to help CMake find your Python environment:
# cmake .. -DCMAKE_PREFIX_PATH=$(python -c "import sys; print(sys.prefix)")
find_package(pybind11 CONFIG REQUIRED)

# FIX 2: Use CUDAToolkit instead of CUDA
# This defines the CUDA::cudart and CUDA::curand targets correctly
# and suppresses the CMP0146 warning.
find_package(CUDAToolkit REQUIRED)

# pybind11_add_module creates a Python module
# It is smart enough to recognize .cpp (for CXX compiler) and .cu (for CUDA compiler)
pybind11_add_module(hubo_solver SHARED
    bindings.cpp
    HUBO.cu
)

# Link our module to the CUDA runtime and CURAND libraries
# This line should now work thanks to FIX 2
target_link_libraries(hubo_solver PRIVATE 
    CUDA::cudart 
    CUDA::curand
)

# Optional: Set C++ and CUDA standards
set_target_properties(hubo_solver PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
)