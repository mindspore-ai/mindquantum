/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDQUANTUM_BACKENDS_QUEST_UTILS_H_
#define MINDQUANTUM_BACKENDS_QUEST_UTILS_H_
#include <memory>

#include "QuEST.h"
#include "QuEST_validation.h"
#include "core/utils.h"
#include "gate/basic_gate.h"
#include "sparse/csrhdmatrix.h"
#ifdef GPUACCELERATED
#    include <cuda_runtime_api.h>
#endif

namespace mindquantum {
namespace quest {
using sparse::CsrHdMatrix;

template <typename T>
inline ComplexMatrixN Dim2Matrix2ComplexMatrixN(const Dim2Matrix<T> &matrix, int n_qubits) {
    auto m = createComplexMatrixN(n_qubits);
    validateMatrixInit(m, __func__);
    int dim = 1 << m.numQubits;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++) {
            m.real[i][j] = std::real(matrix.matrix_[i][j]);
            m.imag[i][j] = std::imag(matrix.matrix_[i][j]);
        }
    return m;
}

template <typename T>
inline ComplexMatrix2 Dim2Matrix2ComplexMatrix2(const Dim2Matrix<T> &matrix) {
    ComplexMatrix2 m;
    int dim = 1 << 1;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++) {
            m.real[i][j] = std::real(matrix.matrix_[i][j]);
            m.imag[i][j] = std::imag(matrix.matrix_[i][j]);
        }
    return m;
}

template <typename T>
inline ComplexMatrix4 Dim2Matrix2ComplexMatrix4(const Dim2Matrix<T> &matrix) {
    ComplexMatrix4 m;
    int dim = 1 << 2;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++) {
            m.real[i][j] = std::real(matrix.matrix_[i][j]);
            m.imag[i][j] = std::imag(matrix.matrix_[i][j]);
        }
    return m;
}

// void destroyComplexMatrix2(ComplexMatrix2 m) {
//   int numRows = 1 << 1;
//   for (int r = 0; r < numRows; r++) {
//     free(m.real[r]);
//     free(m.imag[r]);
//   }
//   free(m.real);
//   free(m.imag);
// }

// void destroyComplexMatrix4(ComplexMatrix4 m) {
//   int numRows = 1 << 2;
//   for (int r = 0; r < numRows; r++) {
//     free(m.real[r]);
//     free(m.imag[r]);
//   }
//   free(m.real);
//   free(m.imag);
// }

inline int *Vec2Intptr(const VT<Index> &vec) {
    int *out = reinterpret_cast<int *>(malloc(sizeof(int) * vec.size()));
    for (size_t i = 0; i < vec.size(); i++) {
        out[i] = static_cast<int>(vec[i]);
    }
    return out;
}
template <typename T>
inline PauliHamil HCast(const Hamiltonian<T> &ham, int n_qubits) {
    PauliHamil quest_ham = createPauliHamil(n_qubits, ham.ham_.size());
    validateHamilParams(quest_ham.numQubits, quest_ham.numSumTerms, __func__);

    int i = 0;
    for (int t = 0; t < quest_ham.numSumTerms; t++) {
        quest_ham.termCoeffs[t] = static_cast<double>(ham.ham_[t].second);
        size_t curr = 0;
        for (size_t q = 0; q < static_cast<size_t>(quest_ham.numQubits); q++) {
            if (curr < ham.ham_[t].first.size()) {
                if (ham.ham_[t].first[curr].first == q) {
                    if (ham.ham_[t].first[curr].second == 'X') {
                        quest_ham.pauliCodes[i] = PAULI_X;
                    } else if (ham.ham_[t].first[curr].second == 'Y') {
                        quest_ham.pauliCodes[i] = PAULI_Y;
                    } else {
                        quest_ham.pauliCodes[i] = PAULI_Z;
                    }
                    curr++;
                } else {
                    quest_ham.pauliCodes[i] = PAULI_I;
                }
            } else {
                quest_ham.pauliCodes[i] = PAULI_I;
            }
            i++;
        }
    }
    return quest_ham;
}

template <typename T>
inline CT<T> Complex2Complex(Complex a) {
    CT<T> res = {a.real, a.imag};
    return res;
}

template <typename T>
inline void Csr_Dot_Vec(std::shared_ptr<CsrHdMatrix<T>> a, Qureg qureg) {
    auto dim = a->dim_;
    auto vec_real = reinterpret_cast<qreal *>(malloc(sizeof(qreal) * dim));
    auto vec_imag = reinterpret_cast<qreal *>(malloc(sizeof(qreal) * dim));
    auto ori_real = reinterpret_cast<qreal *>(malloc(sizeof(qreal) * dim));
    auto ori_imag = reinterpret_cast<qreal *>(malloc(sizeof(qreal) * dim));
    // auto nnz = a->nnz_;
    auto data = a->data_;
    auto indptr = a->indptr_;
    auto indices = a->indices_;
#ifdef GPUACCELERATED
    cudaMemcpy(ori_real, qureg.deviceStateVec.real, dim * sizeof(qreal), cudaMemcpyDeviceToHost);
    cudaMemcpy(ori_imag, qureg.deviceStateVec.imag, dim * sizeof(qreal), cudaMemcpyDeviceToHost);
#else
#    pragma omp parallel for schedule(static)
    for (Index i = 0; i < dim; ++i) {
        ori_real[i] = statevec_getRealAmp(qureg, i);
        ori_imag[i] = statevec_getImagAmp(qureg, i);
    }
#endif
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < dim; i++) {
        CT<T> sum = {0.0, 0.0};
        for (Index j = indptr[i]; j < indptr[i + 1]; j++) {
            sum += data[j] * CT<T>(ori_real[indices[j]], ori_imag[indices[j]]);
        }
        vec_real[i] = std::real(sum);
        vec_imag[i] = std::imag(sum);
    }
    initStateFromAmps(qureg, vec_real, vec_imag);
    free(vec_real);
    free(vec_imag);
    free(ori_real);
    free(ori_imag);
}

template <typename T>
inline void Csr_Dot_Vec(std::shared_ptr<CsrHdMatrix<T>> a, std::shared_ptr<CsrHdMatrix<T>> b, Qureg qureg) {
    auto dim = a->dim_;
    auto vec_real = reinterpret_cast<qreal *>(malloc(sizeof(qreal) * dim));
    auto vec_imag = reinterpret_cast<qreal *>(malloc(sizeof(qreal) * dim));
    auto ori_real = reinterpret_cast<qreal *>(malloc(sizeof(qreal) * dim));
    auto ori_imag = reinterpret_cast<qreal *>(malloc(sizeof(qreal) * dim));
    // auto nnz = a->nnz_;
    auto data = a->data_;
    auto indptr = a->indptr_;
    auto indices = a->indices_;
    auto data_b = b->data_;
    auto indptr_b = b->indptr_;
    auto indices_b = b->indices_;

#ifdef GPUACCELERATED
    cudaMemcpy(ori_real, qureg.deviceStateVec.real, dim * sizeof(qreal), cudaMemcpyDeviceToHost);
    cudaMemcpy(ori_imag, qureg.deviceStateVec.imag, dim * sizeof(qreal), cudaMemcpyDeviceToHost);
#else
#    pragma omp parallel for schedule(static)
    for (Index i = 0; i < dim; ++i) {
        ori_real[i] = statevec_getRealAmp(qureg, i);
        ori_imag[i] = statevec_getImagAmp(qureg, i);
    }
#endif
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < dim; i++) {
        CT<T> sum = {0.0, 0.0};
        for (Index j = indptr[i]; j < indptr[i + 1]; j++) {
            sum += data[j] * CT<T>(ori_real[indices[j]], ori_imag[indices[j]]);
        }
        for (Index j = indptr_b[i]; j < indptr_b[i + 1]; j++) {
            sum += data_b[j] * CT<T>(ori_real[indices_b[j]], ori_imag[indices_b[j]]);
        }
        vec_real[i] = std::real(sum);
        vec_imag[i] = std::imag(sum);
    }
    initStateFromAmps(qureg, vec_real, vec_imag);
    free(vec_real);
    free(vec_imag);
    free(ori_real);
    free(ori_imag);
}

}  // namespace quest
}  // namespace mindquantum
#endif
