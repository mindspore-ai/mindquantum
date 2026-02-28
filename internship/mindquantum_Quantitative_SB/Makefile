# ============================================
# BSB/DSB Makefile - Linux编译
# ============================================

# 编译器
CXX = g++
NVCC = nvcc

# 编译选项
CXX_OPTS = -O3 -std=c++17 -march=native -fopenmp
NVCC_CPU_OPTS = -O3 -std=c++17 -march=native
NVCC_GPU_OPTS = -O3 -std=c++17 -arch=sm_70

# 包含目录
INCLUDE = -I..

# 源文件目录
BSB_DIR = BSB
DSB_DIR = DSB

# 目标文件
BSB_CPU_TARGETS = $(BSB_DIR)/BSB_CPU_fp16 $(BSB_DIR)/BSB_CPU_int8 $(BSB_DIR)/BSB_CPU_fp8
BSB_GPU_TARGETS = $(BSB_DIR)/BSB_GPU_fp16 $(BSB_DIR)/BSB_GPU_int8 $(BSB_DIR)/BSB_GPU_fp8
DSB_CPU_TARGETS = $(DSB_DIR)/DSB_CPU_fp16 $(DSB_DIR)/DSB_CPU_int8 $(DSB_DIR)/DSB_CPU_fp8
DSB_GPU_TARGETS = $(DSB_DIR)/DSB_GPU_fp16 $(DSB_DIR)/DSB_GPU_int8 $(DSB_DIR)/DSB_GPU_fp8

# 默认目标（仅CPU版本）
all: cpu

# 编译所有CPU版本
cpu: $(BSB_CPU_TARGETS) $(DSB_CPU_TARGETS)
	@echo ""
	@echo "============================================"
	@echo "CPU版本编译完成！"
	@echo "============================================"

# 编译所有GPU版本
gpu: check_nvcc $(BSB_CPU_TARGETS) $(DSB_CPU_TARGETS) $(BSB_GPU_TARGETS) $(DSB_GPU_TARGETS)
	@echo ""
	@echo "============================================"
	@echo "所有版本（CPU+GPU）编译完成！"
	@echo "============================================"

# BSB CPU版本
$(BSB_DIR)/BSB_CPU_fp16: $(BSB_DIR)/BSB_CPU_fp16.cu
	@echo "[编译] BSB_CPU_fp16..."
	@$(CXX) $(CXX_OPTS) $(INCLUDE) $^ -o $@
	@echo "  [成功] BSB_CPU_fp16"

$(BSB_DIR)/BSB_CPU_int8: $(BSB_DIR)/BSB_CPU_int8.cu
	@echo "[编译] BSB_CPU_int8..."
	@$(CXX) $(CXX_OPTS) $(INCLUDE) $^ -o $@
	@echo "  [成功] BSB_CPU_int8"

$(BSB_DIR)/BSB_CPU_fp8: $(BSB_DIR)/BSB_CPU_fp8.cu
	@echo "[编译] BSB_CPU_fp8..."
	@$(CXX) $(CXX_OPTS) $(INCLUDE) $^ -o $@
	@echo "  [成功] BSB_CPU_fp8"

# BSB GPU版本
$(BSB_DIR)/BSB_GPU_fp16: $(BSB_DIR)/BSB_GPU_fp16.cu check_nvcc
	@echo "[编译] BSB_GPU_fp16..."
	@$(NVCC) $(NVCC_GPU_OPTS) $(INCLUDE) $^ -o $@ || (echo "  [失败] 检查GPU架构设置"; exit 1)
	@echo "  [成功] BSB_GPU_fp16"

$(BSB_DIR)/BSB_GPU_int8: $(BSB_DIR)/BSB_GPU_int8.cu check_nvcc
	@echo "[编译] BSB_GPU_int8..."
	@$(NVCC) $(NVCC_GPU_OPTS) $(INCLUDE) $^ -o $@ || (echo "  [失败] 检查GPU架构设置"; exit 1)
	@echo "  [成功] BSB_GPU_int8"

$(BSB_DIR)/BSB_GPU_fp8: $(BSB_DIR)/BSB_GPU_fp8.cu check_nvcc
	@echo "[编译] BSB_GPU_fp8..."
	@$(NVCC) $(NVCC_GPU_OPTS) $(INCLUDE) $^ -o $@ || (echo "  [失败] 检查GPU架构设置"; exit 1)
	@echo "  [成功] BSB_GPU_fp8"

# DSB CPU版本
$(DSB_DIR)/DSB_CPU_fp16: $(DSB_DIR)/DSB_CPU_fp16.cu
	@echo "[编译] DSB_CPU_fp16..."
	@$(CXX) $(CXX_OPTS) $(INCLUDE) $^ -o $@
	@echo "  [成功] DSB_CPU_fp16"

$(DSB_DIR)/DSB_CPU_int8: $(DSB_DIR)/DSB_CPU_int8.cu
	@echo "[编译] DSB_CPU_int8..."
	@$(CXX) $(CXX_OPTS) $(INCLUDE) $^ -o $@
	@echo "  [成功] DSB_CPU_int8"

$(DSB_DIR)/DSB_CPU_fp8: $(DSB_DIR)/DSB_CPU_fp8.cu
	@echo "[编译] DSB_CPU_fp8..."
	@$(CXX) $(CXX_OPTS) $(INCLUDE) $^ -o $@
	@echo "  [成功] DSB_CPU_fp8"

# DSB GPU版本
$(DSB_DIR)/DSB_GPU_fp16: $(DSB_DIR)/DSB_GPU_fp16.cu check_nvcc
	@echo "[编译] DSB_GPU_fp16..."
	@$(NVCC) $(NVCC_GPU_OPTS) $(INCLUDE) $^ -o $@ || (echo "  [失败] 检查GPU架构设置"; exit 1)
	@echo "  [成功] DSB_GPU_fp16"

$(DSB_DIR)/DSB_GPU_int8: $(DSB_DIR)/DSB_GPU_int8.cu check_nvcc
	@echo "[编译] DSB_GPU_int8..."
	@$(NVCC) $(NVCC_GPU_OPTS) $(INCLUDE) $^ -o $@ || (echo "  [失败] 检查GPU架构设置"; exit 1)
	@echo "  [成功] DSB_GPU_int8"

$(DSB_DIR)/DSB_GPU_fp8: $(DSB_DIR)/DSB_GPU_fp8.cu check_nvcc
	@echo "[编译] DSB_GPU_fp8..."
	@$(NVCC) $(NVCC_GPU_OPTS) $(INCLUDE) $^ -o $@ || (echo "  [失败] 检查GPU架构设置"; exit 1)
	@echo "  [成功] DSB_GPU_fp8"

# 检查nvcc是否可用
.PHONY: check_nvcc
check_nvcc:
	@command -v nvcc > /dev/null 2>&1 || { \
		echo "错误: nvcc 未安装！GPU版本需要CUDA Toolkit"; \
		echo "请安装CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"; \
		exit 1; \
	}

# 编译单个算法
.PHONY: compile-%
compile-%:
	@if echo "$@" | grep -q "BSB_CPU"; then \
		$(MAKE) $(BSB_DIR)/$(subst compile-,,$@); \
	elif echo "$@" | grep -q "DSB_CPU"; then \
		$(MAKE) $(DSB_DIR)/$(subst compile-,,$@); \
	elif echo "$@" | grep -q "BSB_GPU"; then \
		$(MAKE) check_nvcc; \
		$(MAKE) $(BSB_DIR)/$(subst compile-,,$@); \
	elif echo "$@" | grep -q "DSB_GPU"; then \
		$(MAKE) check_nvcc; \
		$(MAKE) $(DSB_DIR)/$(subst compile-,,$@); \
	else \
		echo "未知的算法: $@"; \
		exit 1; \
	fi

# 清理编译产物
.PHONY: clean
clean:
	@echo "清理编译产物..."
	@rm -f $(BSB_CPU_TARGETS) $(BSB_GPU_TARGETS)
	@rm -f $(DSB_CPU_TARGETS) $(DSB_GPU_TARGETS)
	@echo "清理完成！"

# 清理并重新编译
.PHONY: rebuild
rebuild: clean all

# 显示帮助信息
.PHONY: help
help:
	@echo "============================================"
	@echo "BSB/DSB Makefile 使用指南"
	@echo "============================================"
	@echo ""
	@echo "编译目标:"
	@echo "  make              - 编译所有CPU版本（默认）"
	@echo "  make cpu          - 仅编译CPU版本"
	@echo "  make gpu          - 编译所有版本（CPU + GPU，需要CUDA）"
	@echo ""
	@echo "编译单个算法:"
	@echo "  make compile-BSB_CPU_fp16"
	@echo "  make compile-DSB_GPU_fp8"
	@echo "  make compile-<algorithm_name>"
	@echo ""
	@echo "清理:"
	@echo "  make clean        - 删除编译产物"
	@echo "  make rebuild      - 清理并重新编译"
	@echo ""
	@echo "其他:"
	@echo "  make help         - 显示此帮助信息"
	@echo ""
	@echo "示例:"
	@echo "  make cpu                    # 编译所有CPU版本"
	@echo "  make compile-BSB_CPU_fp16   # 编译单个算法"
	@echo "  make clean                  # 清理编译产物"
	@echo ""
	@echo "============================================"
	@echo "支持的算法:"
	@echo "  BSB:"
	@echo "    - BSB_CPU_fp16, BSB_CPU_int8, BSB_CPU_fp8"
	@echo "    - BSB_GPU_fp16, BSB_GPU_int8, BSB_GPU_fp8"
	@echo "  DSB:"
	@echo "    - DSB_CPU_fp16, DSB_CPU_int8, DSB_CPU_fp8"
	@echo "    - DSB_GPU_fp16, DSB_GPU_int8, DSB_GPU_fp8"
	@echo "============================================"

# 声明伪目标
.PHONY: all cpu gpu clean rebuild help
