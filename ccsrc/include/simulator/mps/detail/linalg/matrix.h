/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_MATRIX_HPP
#define INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_MATRIX_HPP

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mindquantum::sim::mps::detail::linalg {
namespace matrix::detail {
inline size_t CheckedMul(size_t a, size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::runtime_error(std::string("mqmps: size overflow in ") + what + ".");
    }
    return a * b;
}

inline size_t CheckedAdd(size_t a, size_t b, const char* what) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::runtime_error(std::string("mqmps: size overflow in ") + what + ".");
    }
    return a + b;
}
}  // namespace matrix::detail

template <class T>
class MatrixView {
 public:
    using value_type = T;

    MatrixView() = default;
    MatrixView(T* data, size_t rows, size_t cols, size_t ld) : data_(data), rows_(rows), cols_(cols), ld_(ld) {
        if (ld_ < rows_) {
            throw std::runtime_error("mqmps: invalid MatrixView leading dimension (ld < rows).");
        }
        if (rows_ != 0 && cols_ != 0 && data_ == nullptr) {
            throw std::runtime_error("mqmps: null data pointer for non-empty MatrixView.");
        }
    }

    size_t rows() const {
        return rows_;
    }
    size_t cols() const {
        return cols_;
    }
    size_t leading_dim() const {
        return ld_;
    }

    T* data() const {
        return data_;
    }

    bool empty() const {
        return rows_ == 0 || cols_ == 0;
    }

    T& operator()(size_t row, size_t col) const {
        return data_[row + ld_ * col];
    }

    MatrixView<T> block(size_t row0, size_t col0, size_t rows, size_t cols) const {
        if (row0 > rows_ || col0 > cols_ || rows > rows_ - row0 || cols > cols_ - col0) {
            throw std::runtime_error("mqmps: MatrixView::block out of range.");
        }
        if (rows == 0 || cols == 0) {
            return MatrixView<T>(nullptr, rows, cols, ld_);
        }
        const size_t col_offset = matrix::detail::CheckedMul(ld_, col0, "MatrixView::block offset");
        const size_t offset = matrix::detail::CheckedAdd(row0, col_offset, "MatrixView::block offset");
        return MatrixView<T>(data_ + offset, rows, cols, ld_);
    }

 private:
    T* data_ = nullptr;
    size_t rows_ = 0;
    size_t cols_ = 0;
    size_t ld_ = 0;
};

template <class T>
class Matrix {
 public:
    using value_type = T;

    Matrix() = default;
    Matrix(size_t rows, size_t cols) {
        resize(rows, cols);
    }

    size_t rows() const {
        return rows_;
    }
    size_t cols() const {
        return cols_;
    }
    size_t leading_dim() const {
        return rows_;
    }
    size_t size() const {
        return rows_ * cols_;
    }
    size_t storage_size() const {
        return data_.size();
    }
    bool empty() const {
        return size() == 0;
    }

    T* data() {
        return data_.data();
    }
    const T* data() const {
        return data_.data();
    }

    MatrixView<T> view() {
        return MatrixView<T>(data(), rows_, cols_, leading_dim());
    }
    MatrixView<const T> view() const {
        return MatrixView<const T>(data_.data(), rows_, cols_, leading_dim());
    }

    MatrixView<T> block(size_t row0, size_t col0, size_t rows, size_t cols) {
        return view().block(row0, col0, rows, cols);
    }
    MatrixView<const T> block(size_t row0, size_t col0, size_t rows, size_t cols) const {
        return view().block(row0, col0, rows, cols);
    }

    void resize(size_t rows, size_t cols) {
        const size_t required = matrix::detail::CheckedMul(rows, cols, "Matrix::resize");
        rows_ = rows;
        cols_ = cols;
        if (required > data_.size()) {
            data_.resize(required);
        }
    }

    void reserve(size_t rows, size_t cols) {
        const size_t required = matrix::detail::CheckedMul(rows, cols, "Matrix::reserve");
        if (required > data_.size()) {
            data_.resize(required);
        }
    }

    void set_zero() {
        if (size() == 0) {
            return;
        }
        std::fill(data_.begin(), data_.begin() + static_cast<std::ptrdiff_t>(size()), T{});
    }

    void swap(Matrix& other) noexcept {
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        data_.swap(other.data_);
    }

    T& operator()(size_t row, size_t col) {
        return data_[row + leading_dim() * col];
    }
    const T& operator()(size_t row, size_t col) const {
        return data_[row + leading_dim() * col];
    }

 private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    std::vector<T> data_;
};

template <typename T, typename Real>
inline void ScaleByDiagonals(MatrixView<T> a, const std::vector<Real>& left, const std::vector<Real>& right,
                             const char* what = "ScaleByDiagonals") {
    if (a.rows() != left.size() || a.cols() != right.size()) {
        throw std::runtime_error(std::string("mqmps: diag scaling dimension mismatch in ") + what + ": A=("
                                 + std::to_string(a.rows()) + "x" + std::to_string(a.cols()) + "), left="
                                 + std::to_string(left.size()) + ", right=" + std::to_string(right.size()) + ".");
    }

    const size_t rows = a.rows();
    const size_t cols = a.cols();
    const size_t ld = a.leading_dim();
    for (size_t c = 0; c < cols; ++c) {
        const Real s_col = right[c];
        T* col = a.data() + ld * c;
        for (size_t r = 0; r < rows; ++r) {
            col[r] *= left[r] * s_col;
        }
    }
}
}  // namespace mindquantum::sim::mps::detail::linalg

#endif  // INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_MATRIX_HPP
