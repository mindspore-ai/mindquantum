@echo off
REM ============================================
REM Windows 统一编译脚本 - 编译所有BSB和DSB实现
REM ============================================

setlocal EnableDelayedExpansion

echo ============================================
echo BSB/DSB Windows 统一编译脚本
echo ============================================
echo.

REM 检查 nvcc 是否可用
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [错误] nvcc 未找到！请确保已安装 CUDA Toolkit 并添加到 PATH
    echo.
    goto :eof
)

echo [信息] nvcc 已找到
echo.

REM 设置编译选项
REM 使用 RTX 4070 Ti SUPER (compute capability 8.9)
set NVCC_OPTS_CPU=-O3 -arch=native
set NVCC_OPTS_GPU=-O3 -arch=sm_89 -Xcompiler "/std:c++17"
set INCLUDE_DIR=..

REM ========== 编译 BSB ==========
echo ============================================
echo 编译 BSB 算法
echo ============================================
echo.

echo [1/6] 编译 BSB_CPU_fp16.cu ...
nvcc %NVCC_OPTS_CPU% -I%INCLUDE_DIR% BSB\BSB_CPU_fp16.cu -o BSB\BSB_CPU_fp16.exe
if errorlevel 1 (
    echo   [失败] BSB_CPU_fp16 编译失败
) else (
    echo   [成功] BSB_CPU_fp16.exe 生成完成
)

echo [2/6] 编译 BSB_CPU_int8.cu ...
nvcc %NVCC_OPTS_CPU% -I%INCLUDE_DIR% BSB\BSB_CPU_int8.cu -o BSB\BSB_CPU_int8.exe
if errorlevel 1 (
    echo   [失败] BSB_CPU_int8 编译失败
) else (
    echo   [成功] BSB_CPU_int8.exe 生成完成
)

echo [3/6] 编译 BSB_CPU_fp8.cu ...
nvcc %NVCC_OPTS_CPU% -I%INCLUDE_DIR% BSB\BSB_CPU_fp8.cu -o BSB\BSB_CPU_fp8.exe
if errorlevel 1 (
    echo   [失败] BSB_CPU_fp8 编译失败
) else (
    echo   [成功] BSB_CPU_fp8.exe 生成完成
)

echo [4/6] 编译 BSB_GPU_fp16.cu ...
nvcc %NVCC_OPTS_GPU% -I%INCLUDE_DIR% BSB\BSB_GPU_fp16.cu -o BSB\BSB_GPU_fp16.exe
if errorlevel 1 (
    echo   [失败] BSB_GPU_fp16 编译失败
) else (
    echo   [成功] BSB_GPU_fp16.exe 生成完成
)

echo [5/6] 编译 BSB_GPU_int8.cu ...
nvcc %NVCC_OPTS_GPU% -I%INCLUDE_DIR% BSB\BSB_GPU_int8.cu -o BSB\BSB_GPU_int8.exe
if errorlevel 1 (
    echo   [失败] BSB_GPU_int8 编译失败
) else (
    echo   [成功] BSB_GPU_int8.exe 生成完成
)

echo [6/6] 编译 BSB_GPU_fp8.cu ...
nvcc %NVCC_OPTS_GPU% -I%INCLUDE_DIR% BSB\BSB_GPU_fp8.cu -o BSB\BSB_GPU_fp8.exe
if errorlevel 1 (
    echo   [失败] BSB_GPU_fp8 编译失败
) else (
    echo   [成功] BSB_GPU_fp8.exe 生成完成
)

echo.
echo ============================================
echo BSB 编译完成
echo ============================================

REM ========== 编译 DSB ==========
echo.
echo ============================================
echo 编译 DSB 算法
echo ============================================
echo.

echo [7/12] 编译 DSB_CPU_fp16.cu ...
nvcc %NVCC_OPTS_CPU% -I%INCLUDE_DIR% DSB\DSB_CPU_fp16.cu -o DSB\DSB_CPU_fp16.exe
if errorlevel 1 (
    echo   [失败] DSB_CPU_fp16 编译失败
) else (
    echo   [成功] DSB_CPU_fp16.exe 生成完成
)

echo [8/12] 编译 DSB_CPU_int8.cu ...
nvcc %NVCC_OPTS_CPU% -I%INCLUDE_DIR% DSB\DSB_CPU_int8.cu -o DSB\DSB_CPU_int8.exe
if errorlevel 1 (
    echo   [失败] DSB_CPU_int8 编译失败
) else (
    echo   [成功] DSB_CPU_int8.exe 生成完成
)

echo [9/12] 编译 DSB_CPU_fp8.cu ...
nvcc %NVCC_OPTS_CPU% -I%INCLUDE_DIR% DSB\DSB_CPU_fp8.cu -o DSB\DSB_CPU_fp8.exe
if errorlevel 1 (
    echo   [失败] DSB_CPU_fp8 编译失败
) else (
    echo   [成功] DSB_CPU_fp8.exe 生成完成
)

echo [10/12] 编译 DSB_GPU_fp16.cu ...
nvcc %NVCC_OPTS_GPU% -I%INCLUDE_DIR% DSB\DSB_GPU_fp16.cu -o DSB\DSB_GPU_fp16.exe
if errorlevel 1 (
    echo   [失败] DSB_GPU_fp16 编译失败
) else (
    echo   [成功] DSB_GPU_fp16.exe 生成完成
)

echo [11/12] 编译 DSB_GPU_int8.cu ...
nvcc %NVCC_OPTS_GPU% -I%INCLUDE_DIR% DSB\DSB_GPU_int8.cu -o DSB\DSB_GPU_int8.exe
if errorlevel 1 (
    echo   [失败] DSB_GPU_int8 编译失败
) else (
    echo   [成功] DSB_GPU_int8.exe 生成完成
)

echo [12/12] 编译 DSB_GPU_fp8.cu ...
nvcc %NVCC_OPTS_GPU% -I%INCLUDE_DIR% DSB\DSB_GPU_fp8.cu -o DSB\DSB_GPU_fp8.exe
if errorlevel 1 (
    echo   [失败] DSB_GPU_fp8 编译失败
) else (
    echo   [成功] DSB_GPU_fp8.exe 生成完成
)

echo.
echo ============================================
echo 所有编译完成！
echo ============================================
echo.

REM 显示生成的文件
echo 生成的可执行文件:
echo.
echo BSB 目录:
dir /b BSB\*.exe 2>nul
echo.
echo DSB 目录:
dir /b DSB\*.exe 2>nul

echo.
echo 使用方法：
echo   BSB\BSB_CPU_fp16.exe ..\qplib\QPLIB_3565.qplib
echo   DSB\DSB_GPU_fp8.exe ..\qplib\QPLIB_3565.qplib
echo.
