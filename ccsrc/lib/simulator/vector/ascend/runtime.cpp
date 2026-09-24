/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "simulator/vector/ascend/runtime.h"

#include <cmath>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <acl/acl.h>

#include "simulator/vector/ascend/kernel_launch.h"
#include "simulator/vector/ascend/observable.h"
#include "tiling/matrix/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace mindquantum::sim::ascend {
namespace {
void Check(aclError status, const char* operation) {
    if (status != ACL_SUCCESS) {
        const auto* detail = aclGetRecentErrMsg();
        throw std::runtime_error(std::string("mqvector_npu: ") + operation + " failed (" + std::to_string(status) + ")"
                                 + (detail ? std::string(": ") + detail : ""));
    }
}

void Cleanup(aclError status, const char* operation) noexcept {
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "mqvector_npu: %s failed during cleanup (%d)\n", operation, status);
    }
}

// Preserve a caller-owned context, including when allocation/launch throws.
// CANN 8.5 rejects SetCurrentContext(nullptr): a previously unbound thread
// remains bound to our context, but an existing owner's context is restored.
class ContextScope {
 public:
    explicit ContextScope(aclrtContext target = nullptr, void (*check)(aclError, const char*) = Check)
        : target_(target) {
        const auto status = aclrtGetCurrentContext(&previous_);
        if (status != ACL_ERROR_RT_CONTEXT_NULL) {
            check(status, "aclrtGetCurrentContext");
        }
        if (target_ && target_ != previous_) {
            check(aclrtSetCurrentContext(target_), "aclrtSetCurrentContext");
        }
    }
    ~ContextScope() {
        if (previous_ && previous_ != target_) {
            Cleanup(aclrtSetCurrentContext(previous_), "restore caller context");
        }
    }
    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;

 private:
    aclrtContext previous_ = nullptr;
    aclrtContext target_;
};

struct Context {
    aclrtContext handle = nullptr;
    std::string soc;
    unsigned cores;
    unsigned cube_cores;

    explicit Context(int device) {
        static std::once_flag initialized;
        std::call_once(initialized, [] {
            const auto status = aclInit(nullptr);
            // ACL may already be owned by another library in this process.
            if (status != ACL_ERROR_REPEAT_INITIALIZE) {
                Check(status, "aclInit");
            }
        });
        const ContextScope caller;
        Check(aclrtCreateContext(&handle, device), "aclrtCreateContext");
        try {
            const auto* name = aclrtGetSocName();
            if (!name) {
                throw std::runtime_error("aclrtGetSocName returned no device model.");
            }
            soc = name;
            if (soc != MQ_ASCEND_SOC) {
                throw std::runtime_error("Ascend kernel target " MQ_ASCEND_SOC " does not match device " + soc);
            }
            int64_t value = 0;
            Check(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &value), "aclrtGetDeviceInfo");
            if (value <= 0) {
                throw std::runtime_error("Ascend device reports no vector cores.");
            }
            cores = value;
            Check(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_CUBE_CORE_NUM, &value), "aclrtGetDeviceInfo Cube");
            if (value <= 0) {
                throw std::runtime_error("Ascend device reports no Cube cores.");
            }
            cube_cores = value;
        } catch (...) {
            Cleanup(aclrtDestroyContext(handle), "aclrtDestroyContext");
            throw;
        }
    }
    ~Context() {
        // Never reset the whole device or finalize a runtime owned by others.
        Cleanup(aclrtDestroyContext(handle), "aclrtDestroyContext");
    }
};

std::shared_ptr<Context> DeviceContext(int device) {
    static std::mutex mutex;
    static std::map<int, std::weak_ptr<Context>> contexts;
    const std::lock_guard<std::mutex> lock(mutex);
    auto context = contexts[device].lock();
    if (!context) {
        context = std::make_shared<Context>(device);
        contexts[device] = context;
    }
    return context;
}

class Buffer {
 public:
    Buffer(std::shared_ptr<Context> context, size_t bytes) : context_(std::move(context)), bytes_(bytes) {
        const ContextScope scope(context_->handle);
        Check(aclrtMalloc(&data_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~Buffer() {
        // Context errors during unwinding must not replace the original error
        // with std::terminate. Cleanup reports failures without throwing.
        const ContextScope scope(context_->handle, Cleanup);
        Cleanup(aclrtFree(data_), "aclrtFree");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void* Data() const {
        return data_;
    }
    size_t Size() const {
        return bytes_;
    }
    void Upload(const void* source) const {
        Check(aclrtMemcpy(data_, bytes_, source, bytes_, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
    }
    void Download(void* destination) const {
        Check(aclrtMemcpy(destination, bytes_, data_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
    }
    void Swap(Buffer& other) {
        if (context_ != other.context_ || bytes_ != other.bytes_) {
            throw std::logic_error("Cannot swap incompatible Ascend buffers.");
        }
        std::swap(data_, other.data_);
    }

 private:
    std::shared_ptr<Context> context_;
    size_t bytes_;
    void* data_ = nullptr;
};

struct Stream {
    std::shared_ptr<Context> context;
    aclrtStream handle = nullptr;
    explicit Stream(std::shared_ptr<Context> owner) : context(std::move(owner)) {
        const ContextScope scope(context->handle);
        Check(aclrtCreateStream(&handle), "aclrtCreateStream");
    }
    ~Stream() {
        const ContextScope scope(context->handle, Cleanup);
        Cleanup(aclrtSynchronizeStream(handle), "aclrtSynchronizeStream");
        Cleanup(aclrtDestroyStream(handle), "aclrtDestroyStream");
    }
};

// Declare AFTER every temporary buffer/metadata owner used by the transaction.
// On a launch/sync exception drain the stream before those owners unwind. Never
// turn a device failure into a successful result or free queued kernel inputs.
class Transaction {
 public:
    explicit Transaction(aclrtStream stream) : stream_(stream) {
    }
    ~Transaction() {
        if (!finished_) {
            Cleanup(aclrtSynchronizeStream(stream_), "drain failed transaction");
        }
    }
    void Finish(const char* operation) {
        Check(aclrtSynchronizeStream(stream_), operation);
        finished_ = true;
    }

 private:
    aclrtStream stream_;
    bool finished_ = false;
};

class PairLock {
 public:
    PairLock(std::mutex& first, std::mutex& second) : first_(first, std::defer_lock), second_(second, std::defer_lock) {
        if (&first == &second) {
            first_.lock();
        } else {
            std::lock(first_, second_);
        }
    }

 private:
    std::unique_lock<std::mutex> first_, second_;
};

uint64_t Dimension(qbit_t qubits) {
    if (qubits < 0 || qubits > 32) {
        throw std::invalid_argument("mqvector_npu requires 0 <= n_qubits <= 32.");
    }
    return uint64_t{1} << qubits;
}

struct CubeGeometry {
    AscendC::tiling::TCubeTiling tiling{};
    uint32_t workspace_bytes;

    CubeGeometry(const std::shared_ptr<Context>& context, uint32_t dimension, bool adjoint) {
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance(context->soc.c_str());
        if (!platform) {
            throw std::runtime_error("Ascend Cube tiling platform metadata is unavailable.");
        }
        const auto columns = kDenseTileSize / dimension;
        using namespace matmul_tiling;  // NOLINT
        MatmulApiTiling builder(*platform);
        builder.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, false);
        builder.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, false);
        builder.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        const auto states = adjoint ? 2U : 1U;
        builder.SetOrgShape(2 * dimension, states * columns * kDenseBatchPackets, 2 * dimension);
        builder.SetShape(2 * dimension, states * columns * kDenseBatchPackets, 2 * dimension);
        builder.SetFixSplit(2 * dimension, columns, std::min(2 * dimension, uint32_t{64}));
        builder.EnableBias(false);
        builder.SetBufferSpace(-1, -1, -1);
        if (builder.GetTiling(tiling) != 0 || tiling.baseM != 2 * dimension || tiling.baseN != columns) {
            throw std::runtime_error("Ascend contraction requires one packet per FP32 output tile.");
        }
        workspace_bytes = platform->GetLibApiWorkSpaceSize();
        if (!workspace_bytes) {
            throw std::runtime_error("No mixed Matmul workspace size.");
        }
        const auto planes = adjoint ? 10U : 4 * kDensePipelineSlots;
        workspace_bytes += context->cores * planes * kDenseTileSize * kDenseBatchPackets * sizeof(float);
    }
};

class PinnedBuffer {
 public:
    explicit PinnedBuffer(size_t bytes) {
        Check(aclrtMallocHost(&data_, bytes), "aclrtMallocHost");
    }
    ~PinnedBuffer() {
        Cleanup(aclrtFreeHost(data_), "aclrtFreeHost");
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    void* Data() const {
        return data_;
    }

 private:
    void* data_ = nullptr;
};

struct Address {
    size_t offset = 0;
    // Null means the current image, otherwise immutable plan-owned storage.
    const Buffer* resident = nullptr;
};

struct DeviceImage {
    using Key = std::pair<uintptr_t, size_t>;
    using Shape = std::pair<uint32_t, bool>;
    std::map<Key, size_t> locations;
    std::map<Shape, CubeGeometry> geometries;
    std::unique_ptr<Buffer> device;
    std::unique_ptr<PinnedBuffer> host;

    void* At(Address address) const {
        const auto* storage = address.resident ? address.resident : device.get();
        return storage ? static_cast<uint8_t*>(storage->Data()) + address.offset : nullptr;
    }
    size_t Size() const {
        return device ? device->Size() : 0;
    }
    void Upload(aclrtStream stream = nullptr) {
        if (!host) {
            return;
        }
        if (stream) {
            // The enclosing transaction must drain before host/device storage
            // is freed, including if a later upload or launch throws.
            Check(aclrtMemcpyAsync(device->Data(), Size(), host->Data(), Size(), ACL_MEMCPY_HOST_TO_DEVICE, stream),
                  "upload bound metadata");
        } else {
            device->Upload(host->Data());
            host.reset();
        }
    }
};

// Placement, not a growing byte vector. Borrow fragments until Finish, then
// copy directly to one exact-size pinned image. Identity is safe because the
// compiler's immutable constants outlive the resident image and every binding.
// No hash of numeric values, dummy binding, or cross-plan cache is involved.
class Metadata {
 public:
    explicit Metadata(const DeviceImage* resident = nullptr) : resident_(resident) {
    }

    Address Append(const void* source, size_t bytes) {
        if (!bytes) {
            return {};
        }
        const DeviceImage::Key key{reinterpret_cast<uintptr_t>(source), bytes};
        if (resident_) {
            if (const auto found = resident_->locations.find(key); found != resident_->locations.end()) {
                return {found->second, resident_->device.get()};
            }
        }
        const auto [found, inserted] = locations_.emplace(key, size_);
        if (inserted) {
            fragments_.push_back({source, bytes, size_});
            size_ = (size_ + bytes + 31) & ~size_t{31};
        }
        return {found->second, nullptr};
    }

    template <typename value_t>
    Address Append(const std::vector<value_t>& values) {
        return Append(values.data(), values.size() * sizeof(value_t));
    }

    const CubeGeometry& Geometry(const std::shared_ptr<Context>& context, uint32_t dimension, bool adjoint) {
        const DeviceImage::Shape shape{dimension, adjoint};
        if (resident_) {
            if (const auto found = resident_->geometries.find(shape); found != resident_->geometries.end()) {
                return found->second;
            }
        }
        return geometries_.try_emplace(shape, context, dimension, adjoint).first->second;
    }

    std::unique_ptr<DeviceImage> Finish(const std::shared_ptr<Context>& context) {
        const ContextScope scope(context->handle);
        auto image = std::make_unique<DeviceImage>();
        if (size_) {
            image->device = std::make_unique<Buffer>(context, size_);
            image->host = std::make_unique<PinnedBuffer>(size_);
            auto* destination = static_cast<uint8_t*>(image->host->Data());
            for (const auto& fragment : fragments_) {
                std::memcpy(destination + fragment.offset, fragment.source, fragment.bytes);
                const auto end = fragment.offset + fragment.bytes;
                std::memset(destination + end, 0, ((end + 31) & ~size_t{31}) - end);
            }
        }
        image->locations = std::move(locations_);
        image->geometries = std::move(geometries_);
        return image;
    }

 private:
    struct Fragment {
        const void* source;
        size_t bytes, offset;
    };
    const DeviceImage* resident_;
    size_t size_ = 0;
    std::vector<Fragment> fragments_;
    std::map<DeviceImage::Key, size_t> locations_;
    std::map<DeviceImage::Shape, CubeGeometry> geometries_;
};

uint32_t ReductionBlocks(const Context& context, uint64_t plane) {
    return static_cast<uint32_t>(std::min<uint64_t>(std::min(context.cores, kObservableMaxBlocks),
                                                    (plane + kObservableTileSize - 1) / kObservableTileSize));
}

std::complex<double> SumPartials(const ReductionPartial* partials, size_t count) {
    std::complex<double> result{};
    for (size_t i = 0; i < count; ++i) {
        result += std::complex<double>{static_cast<double>(partials[i].real) + partials[i].real_correction,
                                       static_cast<double>(partials[i].imag) + partials[i].imag_correction};
    }
    return result;
}

void ValidateMass(double mass) {
    if (!std::isfinite(mass) || mass <= 0) {
        throw std::invalid_argument("Ascend measurement/sampling requires a positive finite probability mass.");
    }
}

struct CubeTile {
    DensePacket packet;
    Address matrix, config, offsets, indices;
    uint32_t workspace_bytes;
    CubeTile(const std::shared_ptr<Context>& context, const DenseProgram& source, Metadata& metadata,
             bool adjoint = false)
        : packet(source.layout->packet)
        , matrix(metadata.Append(source.coefficients))
        , offsets(metadata.Append(source.layout->offsets))
        , indices(metadata.Append(source.layout->indices)) {
        const auto& geometry = metadata.Geometry(context, source.layout->dimension, adjoint);
        workspace_bytes = geometry.workspace_bytes;
        config = metadata.Append(&geometry.tiling, sizeof(geometry.tiling));
    }
    size_t InstructionCount() const {
        return 1;
    }
    aclError Launch(const Context& context, aclrtStream stream, void* state, uint64_t plane,
                    const DeviceImage& metadata, void* workspace) const {
        constexpr uint32_t packets_per_block = 2 * kDenseBatchPackets;
        const auto blocks = static_cast<uint32_t>(
            std::min<uint64_t>(context.cube_cores, (plane / packet.size + packets_per_block - 1) / packets_per_block));
        return LaunchDense(blocks, stream, state, metadata.At(matrix), metadata.At(offsets), metadata.At(indices),
                           metadata.At(config), workspace, packet);
    }
};
}  // namespace

struct Executable::Impl {
    struct Tile {
        uint64_t external_mask;
        uint64_t tile_count;
        uint32_t size, run, count;
        Address instructions, offsets, pool;
        Tile(const std::shared_ptr<Context>&, const TileProgram& source, Metadata& metadata)
            : external_mask(source.external_mask)
            , tile_count(source.tile_count)
            , size(source.size)
            , run(source.run)
            , count(source.instructions.size())
            , instructions(metadata.Append(source.instructions))
            , offsets(metadata.Append(source.offsets))
            , pool(metadata.Append(*source.data)) {
        }
        size_t InstructionCount() const {
            return count;
        }
        aclError Launch(const Context& context, aclrtStream stream, void* state, uint64_t plane,
                        const DeviceImage& metadata, void*) const {
            if (!count) {
                return ACL_SUCCESS;
            }
            const auto blocks = static_cast<uint32_t>(std::min<uint64_t>(context.cores, tile_count));
            return LaunchTile(blocks, stream, state, metadata.At(instructions), metadata.At(pool), metadata.At(offsets),
                              plane, external_mask, tile_count, size, run, count);
        }
    };
    using Dense = CubeTile;
    std::shared_ptr<Context> context;
    qbit_t n_qubits;
    std::unique_ptr<DeviceImage> metadata;
    std::vector<std::variant<Tile, Dense>> tiles;
    uint32_t workspace_bytes = 0;

    Impl(std::shared_ptr<Context> owner, const Program& program, const DeviceImage* resident = nullptr)
        : context(std::move(owner)), n_qubits(program.n_qubits) {
        Metadata image(resident);
        for (const auto& tile : program.tiles) {
            std::visit(
                [&](const auto& source) {
                    if (!source) {
                        return;  // parameter-dependent Vector slot in Constants()
                    }
                    using operation_t
                        = std::conditional_t<std::is_same_v<std::decay_t<decltype(*source)>, TileProgram>, Tile, Dense>;
                    tiles.emplace_back(std::in_place_type<operation_t>, context, *source, image);
                    if constexpr (std::is_same_v<operation_t, Dense>) {
                        workspace_bytes = std::max(workspace_bytes, std::get<Dense>(tiles.back()).workspace_bytes);
                    }
                },
                tile);
        }
        metadata = image.Finish(context);
    }

    void Launch(aclrtStream stream, void* state, uint64_t plane, void* workspace) const {
        for (const auto& tile : tiles) {
            std::visit(
                [&](const auto& operation) {
                    Check(operation.Launch(*context, stream, state, plane, *metadata, workspace), "launch tile");
                },
                tile);
        }
    }
};

struct Observable::Impl {
    std::shared_ptr<Context> context;
    ObservableLayout layout;
    std::unique_ptr<DeviceImage> metadata;
    std::unique_ptr<Buffer> diagonal;
    uint64_t diagonal_terms = 0;
    Address groups, terms, indices, signs, rows, columns, values;

    Impl(std::shared_ptr<Context> owner, const ObservableProgram& program)
        : context(std::move(owner)), layout(program.layout()) {
        if (layout.kind == kObservablePauli && program.groups().size() == 1 && program.groups().front().flip_mask == 0
            && program.terms().size() > 1) {
            double magnitude = 0;
            bool real = true;
            for (const auto& term : program.terms()) {
                magnitude += std::abs(static_cast<double>(term.coefficient.real));
                real = real && term.coefficient.imag == 0;
            }
            if (real && magnitude <= std::numeric_limits<float>::max()) {
                diagonal_terms = program.terms().size();
            }
        }
        Metadata image;
        groups = image.Append(program.groups());
        terms = image.Append(program.terms());
        indices = image.Append(program.gather_indices());
        signs = image.Append(program.signs());
        rows = image.Append(program.row_offsets());
        columns = image.Append(program.columns());
        values = image.Append(program.values());
        metadata = image.Finish(context);
        metadata->Upload();
    }

    // Called only before publishing a gradient plan. The resulting cache is
    // immutable, so independent evaluations can safely share it.
    void CacheDiagonal() {
        const ContextScope scope(context->handle);
        auto cache = std::make_unique<Buffer>(context, layout.plane * sizeof(float));
        Stream stream(context);
        Transaction transaction(stream.handle);
        Check(LaunchCompileDiagonal(ReductionBlocks(*context, layout.plane), stream.handle, metadata->At(terms),
                                    metadata->At(signs), cache->Data(), diagonal_terms, layout),
              "compile diagonal observable");
        transaction.Finish("cache diagonal observable");
        diagonal = std::move(cache);
        layout.kind = kObservableDiagonal;
    }

    void Validate(const std::shared_ptr<Context>& owner, uint64_t dimension) const {
        if (context != owner || layout.dimension != dimension) {
            throw std::invalid_argument("Observable requires the same Ascend device and qubit count.");
        }
    }

    void Launch(aclrtStream stream, void* ket, void* bra, void* output, void* partials) const {
        Check(LaunchObservable(ReductionBlocks(*context, layout.plane), stream, ket, bra, output, metadata->At(groups),
                               metadata->At(terms), metadata->At(indices), metadata->At(signs), metadata->At(rows),
                               metadata->At(columns), diagonal ? diagonal->Data() : metadata->At(values), partials,
                               layout),
              "launch observable");
    }
};

namespace {
// A bound adjoint has one immutable metadata upload, reused for every Hamiltonian.
struct AdjointExecutable {
    struct Tile {
        uint64_t external_mask, tile_count;
        uint32_t size, run, count;
        Address instructions, offsets, steps, derivatives, targets, pool;
    };
    struct Dense {
        CubeTile cube;
        Address derivatives, outputs, inverse;
        uint32_t count;
        Dense(const std::shared_ptr<Context>& context, const DenseAdjointProgram& source, Metadata& metadata)
            : cube(context, source.evolution, metadata, true)
            , derivatives(metadata.Append(source.derivatives))
            , outputs(metadata.Append(source.outputs))
            , inverse(metadata.Append(source.evolution.layout->inverse_indices))
            , count(source.outputs.size()) {
        }
    };
    std::unique_ptr<DeviceImage> metadata;
    std::vector<std::variant<Tile, Dense>> tiles;
    uint32_t blocks, cube_blocks, output_size;
    uint32_t workspace_bytes = 0;
    uint64_t plane;

    AdjointExecutable(const std::shared_ptr<Context>& context, const AdjointProgram& program, qbit_t qubits,
                      const MST<size_t>& parameter_map, const DeviceImage* resident = nullptr)
        : blocks(context->cores)
        , cube_blocks(context->cube_cores)
        , output_size(program.output_size)
        , plane(std::max(Dimension(qubits), uint64_t{8})) {
        if (program.n_qubits != qubits || program.output_size != parameter_map.size() + 1
            || program.parameter_map != parameter_map || program.tiles.empty()) {
            throw std::invalid_argument("Incompatible Ascend adjoint program.");
        }
        Metadata image(resident);
        for (const auto& entry : program.tiles) {
            std::visit(
                [&](const auto& pointer) {
                    if (!pointer) {
                        return;
                    }
                    const auto& source = *pointer;
                    if constexpr (std::is_same_v<std::decay_t<decltype(source)>, AdjointTileProgram>) {
                        tiles.emplace_back(Tile{source.external_mask, source.tile_count, source.size, source.run,
                                                static_cast<uint32_t>(source.steps.size()),
                                                image.Append(source.instructions), image.Append(source.offsets),
                                                image.Append(source.steps), image.Append(source.derivatives),
                                                image.Append(source.targets), image.Append(*source.data)});
                    } else {
                        tiles.emplace_back(std::in_place_type<Dense>, context, source, image);
                        workspace_bytes = std::max(workspace_bytes, std::get<Dense>(tiles.back()).cube.workspace_bytes);
                    }
                },
                entry);
        }
        metadata = image.Finish(context);
    }

    void Launch(aclrtStream stream, void* shared, void* costate, const Buffer& partials, void* workspace) const {
        Check(aclrtMemsetAsync(partials.Data(), partials.Size(), 0, AdjointPartialBytes(blocks, output_size), stream),
              "zero adjoint partials");
        bool overlap = true;
        for (const auto& entry : tiles) {
            // DO NOT reduce the block count for smaller tiles: accumulated slots
            // retain their block ownership throughout the complete sweep.
            std::visit(
                [&](const auto& tile) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(tile)>, Tile>) {
                        Check(LaunchAdjointTile(blocks, stream, shared, costate, metadata->At(tile.instructions),
                                                metadata->At(tile.pool), metadata->At(tile.offsets),
                                                metadata->At(tile.steps), metadata->At(tile.derivatives),
                                                metadata->At(tile.targets), partials.Data(), plane, tile.external_mask,
                                                tile.tile_count, tile.size, tile.run, tile.count, output_size, overlap),
                              "launch adjoint tile");
                    } else {
                        const auto& cube = tile.cube;
                        Check(LaunchDenseAdjoint(cube_blocks, stream, shared, costate, metadata->At(cube.matrix),
                                                 metadata->At(tile.derivatives), metadata->At(tile.outputs),
                                                 metadata->At(cube.offsets), metadata->At(cube.indices),
                                                 metadata->At(tile.inverse), metadata->At(cube.config), workspace,
                                                 partials.Data(), cube.packet, tile.count, output_size, overlap),
                              "launch dense adjoint");
                    }
                },
                entry);
            overlap = false;
        }
    }

    std::vector<std::complex<double>> Read(const Buffer& partials) const {
        std::vector<float> host(partials.Size() / sizeof(float));
        partials.Download(host.data());
        std::vector<std::complex<double>> result(output_size);
        for (uint32_t block = 0; block < blocks; ++block) {
            for (uint32_t column = 0; column < output_size; ++column) {
                const auto offset = (static_cast<size_t>(block) * output_size + column) * kGradientPartialStride;
                result[column] += std::complex<double>{static_cast<double>(host[offset]) - host[offset + 2],
                                                       static_cast<double>(host[offset + 1]) - host[offset + 3]};
            }
        }
        return result;
    }
};
}  // namespace

struct State::Impl {
    qbit_t n_qubits;
    uint64_t dimension, plane;
    std::shared_ptr<Context> context;
    Buffer state;
    // Mutable communication/staging belongs to a serialized state stream,
    // never to an executable that independent states may replay concurrently.
    std::unique_ptr<Buffer> workspace;
    Stream stream;  // destroyed/synchronized before the buffers
    mutable std::mutex mutex;

    Impl(qbit_t qubits, std::shared_ptr<Context> owner)
        : n_qubits(qubits)
        , dimension(Dimension(qubits))
        , plane(std::max(dimension, uint64_t{8}))
        , context(std::move(owner))
        , state(context, 2 * plane * sizeof(float))
        , stream(context) {
    }

    void Validate(const Impl& other) const {
        if (context != other.context || n_qubits != other.n_qubits) {
            throw std::invalid_argument("States require the same Ascend device and qubit count.");
        }
    }

    std::complex<double> Reduce(void* bra, uint64_t mask = 0, uint64_t value = 0) const {
        if ((mask & ~(dimension - 1)) || (value & ~mask)) {
            throw std::invalid_argument("Norm mask must be in range and value must be a subset of mask.");
        }
        const auto blocks = ReductionBlocks(*context, plane);
        Buffer partials(context, blocks * sizeof(ReductionPartial));
        std::vector<ReductionPartial> host(blocks);
        Transaction transaction(stream.handle);
        Check(LaunchInnerProduct(blocks, stream.handle, bra, state.Data(), partials.Data(), dimension, mask, value),
              "launch inner product");
        transaction.Finish("reduce state");
        partials.Download(host.data());
        return SumPartials(host.data(), host.size());
    }
};

Executable::Executable(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}
Executable::~Executable() = default;
size_t Executable::TileCount() const {
    return impl_->tiles.size();
}
size_t Executable::CubeTileCount() const {
    return std::count_if(impl_->tiles.begin(), impl_->tiles.end(),
                         [](const auto& tile) { return std::holds_alternative<Impl::Dense>(tile); });
}
size_t Executable::InstructionCount() const {
    size_t count = 0;
    for (const auto& tile : impl_->tiles) {
        count += std::visit([](const auto& operation) { return operation.InstructionCount(); }, tile);
    }
    return count;
}
size_t Executable::MetadataBytes() const {
    return impl_->metadata ? impl_->metadata->Size() : 0;
}

State::State(qbit_t n_qubits, int device) {
    Dimension(n_qubits);  // reject invalid shifts before touching the runtime
    impl_ = std::make_unique<Impl>(n_qubits, DeviceContext(device));
    Reset();
}
State::State(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}
State::~State() = default;

std::shared_ptr<Executable> State::Prepare(const Program& program) const {
    if (program.n_qubits != impl_->n_qubits) {
        throw std::invalid_argument("Compiled circuit and Ascend state qubit counts differ.");
    }
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    auto executable = std::make_unique<Executable::Impl>(impl_->context, program);
    executable->metadata->Upload();
    return std::shared_ptr<Executable>(new Executable(std::move(executable)));
}

void State::Apply(const Executable& executable, unsigned repetitions) {
    if (executable.impl_->n_qubits != impl_->n_qubits || executable.impl_->context != impl_->context) {
        throw std::invalid_argument("Compiled circuit requires the same Ascend device and qubit count.");
    }
    if (repetitions == 0) {
        throw std::invalid_argument("Execution repetitions must be positive.");
    }
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    const auto bytes = executable.impl_->workspace_bytes;
    if (bytes && (!impl_->workspace || impl_->workspace->Size() < bytes)) {
        impl_->workspace = std::make_unique<Buffer>(impl_->context, bytes);
    }
    Transaction transaction(impl_->stream.handle);
    for (unsigned repeat = 0; repeat < repetitions; ++repeat) {
        executable.impl_->Launch(impl_->stream.handle, impl_->state.Data(), impl_->plane,
                                 impl_->workspace ? impl_->workspace->Data() : nullptr);
    }
    // A circuit is one stream transaction, not one synchronization per gate.
    transaction.Finish("execute circuit");
}

void State::Reset() {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    const auto blocks = static_cast<uint32_t>(
        std::min<uint64_t>(impl_->context->cores, (2 * impl_->plane + kMaxTileSize - 1) / kMaxTileSize));
    Transaction transaction(impl_->stream.handle);
    Check(LaunchReset(blocks, impl_->stream.handle, impl_->state.Data(), impl_->plane), "launch reset");
    transaction.Finish("reset state");
}

void State::Set(const std::complex<float>* data, size_t size) {
    if (size != impl_->dimension) {
        throw std::invalid_argument("Ascend state length does not match its qubit count.");
    }
    std::vector<float> planes(2 * impl_->plane, 0);
    for (size_t i = 0; i < size; ++i) {
        planes[i] = data[i].real();
        planes[impl_->plane + i] = data[i].imag();
    }
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    impl_->state.Upload(planes.data());
}

std::vector<std::complex<float>> State::Get() const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    std::vector<float> planes(2 * impl_->plane);
    Check(aclrtMemcpy(planes.data(), impl_->state.Size(), impl_->state.Data(), impl_->state.Size(),
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "aclrtMemcpy D2H");
    std::vector<std::complex<float>> result(impl_->dimension);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = {planes[i], planes[impl_->plane + i]};
    }
    return result;
}

std::unique_ptr<State> State::Copy() const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    auto copy = std::make_unique<Impl>(impl_->n_qubits, impl_->context);
    Check(aclrtMemcpy(copy->state.Data(), copy->state.Size(), impl_->state.Data(), impl_->state.Size(),
                      ACL_MEMCPY_DEVICE_TO_DEVICE),
          "aclrtMemcpy D2D");
    return std::unique_ptr<State>(new State(std::move(copy)));
}

void State::CopyFrom(const State& other) {
    impl_->Validate(*other.impl_);
    const PairLock lock(impl_->mutex, other.impl_->mutex);
    if (this == &other) {
        return;
    }
    const ContextScope scope(impl_->context->handle);
    Check(aclrtMemcpy(impl_->state.Data(), impl_->state.Size(), other.impl_->state.Data(), other.impl_->state.Size(),
                      ACL_MEMCPY_DEVICE_TO_DEVICE),
          "copy state D2D");
}

Observable::Observable(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}
Observable::~Observable() = default;

std::shared_ptr<Observable> State::PrepareObservable(const Hamiltonian<float>& hamiltonian) const {
    const auto program = ObservableProgram::Compile(hamiltonian, impl_->n_qubits);
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    return std::shared_ptr<Observable>(new Observable(std::make_unique<Observable::Impl>(impl_->context, program)));
}

std::complex<double> State::Expectation(const Observable& observable, const State* bra) const {
    if (!bra) {
        bra = this;
    }
    impl_->Validate(*bra->impl_);
    observable.impl_->Validate(impl_->context, impl_->dimension);
    const PairLock lock(impl_->mutex, bra->impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    const auto blocks = ReductionBlocks(*impl_->context, impl_->plane);
    Buffer partials(impl_->context, blocks * sizeof(ReductionPartial));
    std::vector<ReductionPartial> host(blocks);
    Transaction transaction(impl_->stream.handle);
    // H|ket> exists only in a kernel-local tile, never a full device state.
    observable.impl_->Launch(impl_->stream.handle, impl_->state.Data(), bra->impl_->state.Data(), nullptr,
                             partials.Data());
    transaction.Finish("expectation");
    partials.Download(host.data());
    return SumPartials(host.data(), host.size());
}

void State::ApplyHamiltonian(const Observable& observable) {
    observable.impl_->Validate(impl_->context, impl_->dimension);
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    Buffer output(impl_->context, impl_->state.Size());
    Transaction transaction(impl_->stream.handle);
    observable.impl_->Launch(impl_->stream.handle, impl_->state.Data(), nullptr, output.Data(), nullptr);
    transaction.Finish("apply Hamiltonian");
    impl_->state.Swap(output);  // commit only after successful execution
}

std::complex<double> State::Dot(const State& ket) const {
    impl_->Validate(*ket.impl_);
    const PairLock lock(impl_->mutex, ket.impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    return ket.impl_->Reduce(impl_->state.Data());
}

double State::Norm(uint64_t mask, uint64_t value) const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    return impl_->Reduce(nullptr, mask, value).real();
}

std::vector<std::vector<std::complex<double>>> State::ReducedDensity(const qbits_t& kept_qubits) const {
    uint64_t mask = 0;
    for (auto qubit : kept_qubits) {
        if (qubit < 0 || qubit >= impl_->n_qubits || (mask & (uint64_t{1} << qubit))) {
            throw std::invalid_argument("Reduced density qubits must be distinct and inside the state.");
        }
        mask |= uint64_t{1} << qubit;
    }
    const auto program = CompileDensity(impl_->n_qubits, mask, impl_->context->cores);
    const auto& layout = program.layout;
    const auto records = layout.output_dimension * layout.output_dimension * layout.shards;
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    Metadata image;
    const auto offsets = image.Append(program.offsets);
    const auto indices = image.Append(program.indices);
    auto metadata = image.Finish(impl_->context);
    Buffer partials(impl_->context, records * sizeof(ReductionPartial));
    std::vector<ReductionPartial> host(records);
    Transaction transaction(impl_->stream.handle);
    metadata->Upload(impl_->stream.handle);
    Check(LaunchDensity(static_cast<uint32_t>(std::min<uint64_t>(records, impl_->context->cores)), impl_->stream.handle,
                        impl_->state.Data(), metadata->At(offsets), metadata->At(indices), partials.Data(), layout),
          "launch reduced density matrix");
    transaction.Finish("reduce density matrix");
    partials.Download(host.data());
    std::vector<std::vector<std::complex<double>>> result(layout.output_dimension,
                                                          std::vector<std::complex<double>>(layout.output_dimension));
    for (size_t row = 0; row < result.size(); ++row) {
        for (size_t column = 0; column < result.size(); ++column) {
            const auto offset = (row * result.size() + column) * layout.shards;
            result[row][column] = SumPartials(host.data() + offset, layout.shards);
        }
    }
    return result;
}

void State::Scale(double factor) {
    const auto scalar = static_cast<float>(factor);
    if (!std::isfinite(factor) || !std::isfinite(scalar)) {
        throw std::invalid_argument("Ascend scale requires a finite float32 factor.");
    }
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    Transaction transaction(impl_->stream.handle);
    Check(LaunchScale(ReductionBlocks(*impl_->context, impl_->plane), impl_->stream.handle, impl_->state.Data(),
                      impl_->dimension, scalar),
          "launch scale");
    transaction.Finish("scale state");
}

int State::Measure(qbit_t qubit, double draw, int reset_to) {
    if (qubit < 0 || qubit >= impl_->n_qubits || !std::isfinite(draw) || draw < 0 || draw >= 1 || reset_to < -1
        || reset_to > 1) {
        throw std::invalid_argument("Measure requires an in-range qubit, 0 <= draw < 1, and reset_to in {-1,0,1}.");
    }
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const ContextScope scope(impl_->context->handle);
    const auto bit = uint64_t{1} << qubit;
    const auto blocks = ReductionBlocks(*impl_->context, impl_->plane);
    Buffer partials(impl_->context, 2 * blocks * sizeof(ReductionPartial));
    std::vector<ReductionPartial> host(2 * blocks);
    {
        Transaction transaction(impl_->stream.handle);
        // Reduce both branches directly: subtracting nearly equal total/branch
        // masses would lose the normalization of a rare outcome.
        for (unsigned outcome = 0; outcome < 2; ++outcome) {
            auto* output = static_cast<uint8_t*>(partials.Data()) + outcome * blocks * sizeof(ReductionPartial);
            Check(LaunchInnerProduct(blocks, impl_->stream.handle, nullptr, impl_->state.Data(), output,
                                     impl_->dimension, bit, outcome ? bit : 0),
                  "launch measurement probability");
        }
        transaction.Finish("measurement probabilities");
    }
    partials.Download(host.data());
    const double masses[] = {SumPartials(host.data(), blocks).real(), SumPartials(host.data() + blocks, blocks).real()};
    const double total = masses[0] + masses[1];
    ValidateMass(total);
    if (!std::isfinite(masses[0]) || !std::isfinite(masses[1]) || masses[0] < 0 || masses[1] < 0) {
        throw std::invalid_argument("Invalid Ascend measurement branch probabilities.");
    }
    const int outcome = draw < masses[1] / total ? 1 : 0;
    ValidateMass(masses[outcome]);
    const auto inverse_norm = static_cast<float>(1 / std::sqrt(masses[outcome]));
    if (!std::isfinite(inverse_norm) || inverse_norm <= 0) {
        throw std::invalid_argument("Ascend measurement normalization is not representable in float32.");
    }
    Transaction transaction(impl_->stream.handle);
    Check(LaunchCollapse(blocks, impl_->stream.handle, impl_->state.Data(), impl_->dimension, qubit, outcome, reset_to,
                         inverse_norm),
          "launch collapse");
    transaction.Finish("collapse state");
    return outcome;
}

std::vector<uint64_t> State::Sample(uint64_t shots, uint64_t seed) const {
    if (shots > std::numeric_limits<size_t>::max() / sizeof(SamplingResult)) {
        throw std::invalid_argument("Ascend sampling shot count is too large.");
    }
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!shots) {
        return {};
    }
    const ContextScope scope(impl_->context->handle);
    const auto layout = MakeSamplingLayout(impl_->n_qubits, ReductionBlocks(*impl_->context, impl_->plane));
    Buffer tree(impl_->context, 2 * uint64_t{layout.bins} * layout.tiles_per_bin * sizeof(float));
    Buffer totals(impl_->context, layout.bins * sizeof(ReductionPartial));
    std::vector<ReductionPartial> host_totals(layout.bins);
    {
        Transaction transaction(impl_->stream.handle);
        Check(LaunchSamplingTree(layout.bins, impl_->stream.handle, impl_->state.Data(), tree.Data(), totals.Data(),
                                 layout),
              "launch sampling tree");
        transaction.Finish("build sampling tree");
    }
    totals.Download(host_totals.data());
    std::vector<double> masses;
    std::vector<uint32_t> positive_bins;
    for (uint32_t bin = 0; bin < layout.bins; ++bin) {
        const auto mass = SumPartials(host_totals.data() + bin, 1).real();
        if (!std::isfinite(mass) || mass < 0) {
            throw std::invalid_argument("Invalid Ascend sampling bin probability.");
        }
        if (mass > 0) {
            masses.push_back(mass);
            positive_bins.push_back(bin);
        }
    }
    ValidateMass(SumPartials(host_totals.data(), layout.bins).real());
    std::mt19937_64 rng(seed);
    std::discrete_distribution<uint32_t> choose_bin(masses.begin(), masses.end());
    std::vector<SamplingRequest> requests(shots);
    for (auto& request : requests) {
        // Exclude empty bins even at an RNG/CDF boundary exactly equal to zero.
        request.bin = positive_bins[choose_bin(rng)];
        request.fraction = std::min(static_cast<float>(std::generate_canonical<double, 53>(rng)),
                                    std::nextafter(1.0F, 0.0F));
    }
    Buffer queries(impl_->context, requests.size() * sizeof(SamplingRequest));
    Buffer answers(impl_->context, shots * sizeof(SamplingResult));
    std::vector<SamplingResult> host_answers(shots);
    queries.Upload(requests.data());
    Transaction transaction(impl_->stream.handle);
    Check(LaunchSample(std::min<uint64_t>(impl_->context->cores, std::min<uint64_t>(kObservableMaxBlocks, shots)),
                       impl_->stream.handle, impl_->state.Data(), tree.Data(), queries.Data(), answers.Data(), shots,
                       layout),
          "launch terminal sampling");
    transaction.Finish("terminal sampling");
    answers.Download(host_answers.data());
    std::vector<uint64_t> result;
    result.reserve(shots);
    for (const auto& answer : host_answers) {
        if (answer.index >= impl_->dimension) {
            throw std::runtime_error("Ascend terminal sampling returned an invalid basis index.");
        }
        result.push_back(answer.index);
    }
    return result;
}

std::string State::SocName() const {
    return impl_->context->soc;
}
unsigned State::VectorCores() const {
    return impl_->context->cores;
}

struct GradientPlan::Impl {
    std::shared_ptr<Context> context;
    qbit_t n_qubits;
    bool hermitian;
    MST<size_t> parameter_map;
    std::set<std::string> required_parameters;
    std::unique_ptr<ForwardPlan> right_forward, left_forward;
    std::unique_ptr<AdjointPlan> right_adjoint, left_adjoint;
    // Destroy device placement before its immutable host source programs.
    std::unique_ptr<DeviceImage> right_forward_data, left_forward_data, right_adjoint_data, left_adjoint_data;
    std::vector<std::shared_ptr<Observable>> observables, daggers;
};

GradientPlan::GradientPlan(const State& owner, const std::vector<Hamiltonian<float>>& hams,
                           const std::vector<Hamiltonian<float>>& hams_dagger,
                           const std::vector<std::shared_ptr<BasicGate>>& right_gates,
                           const std::vector<std::shared_ptr<BasicGate>>& left_gates,
                           const std::vector<std::string>& parameter_names, bool hermitian, unsigned tile_qubits,
                           unsigned fusion_qubits, unsigned cube_min_gates) {
    if ((!hermitian || !hams_dagger.empty()) && hams_dagger.size() != hams.size()) {
        throw std::invalid_argument("Gradient requires one H-dagger for each Hamiltonian.");
    }
    if (parameter_names.size() >= std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Too many Ascend gradient parameters.");
    }
    auto plan = std::make_unique<Impl>();
    plan->context = owner.impl_->context;
    plan->n_qubits = owner.impl_->n_qubits;
    plan->hermitian = hermitian;
    for (size_t column = 0; column < parameter_names.size(); ++column) {
        if (!plan->parameter_map.emplace(parameter_names[column], column).second) {
            throw std::invalid_argument("Ascend gradient parameter names must be distinct.");
        }
        plan->required_parameters.insert(parameter_names[column]);
    }
    auto validate_gates = [&](const auto& gates) {
        for (const auto& gate : gates) {
            if (!gate) {
                throw std::invalid_argument("Null gate in Ascend gradient circuit.");
            }
            if (const auto* parameterized = dynamic_cast<const Parameterizable*>(gate.get())) {
                for (const auto& coefficient : parameterized->GetCoeffs()) {
                    ValidateResolver(coefficient);
                    for (const auto& [name, value] : coefficient.data_) {
                        plan->required_parameters.insert(name);
                        if (!coefficient.NoGradContains(name) && !plan->parameter_map.count(name)) {
                            throw std::invalid_argument("Gradient parameter '" + name
                                                        + "' is missing from the output map.");
                        }
                    }
                }
            }
        }
    };
    validate_gates(right_gates);
    if (!hermitian) {
        validate_gates(left_gates);
    }
    plan->right_forward = std::make_unique<ForwardPlan>(right_gates, plan->n_qubits, tile_qubits, fusion_qubits,
                                                        cube_min_gates);
    plan->right_adjoint = std::make_unique<AdjointPlan>(right_gates, plan->parameter_map, plan->n_qubits, tile_qubits,
                                                        fusion_qubits, cube_min_gates);
    if (!hermitian) {
        plan->left_forward = std::make_unique<ForwardPlan>(left_gates, plan->n_qubits, tile_qubits, fusion_qubits,
                                                           cube_min_gates);
        plan->left_adjoint = std::make_unique<AdjointPlan>(left_gates, plan->parameter_map, plan->n_qubits, tile_qubits,
                                                           fusion_qubits, cube_min_gates);
    }
    // Place compile-time data once. The same lowering handles constant and
    // bound programs, but no parameter-dependent gate is evaluated here.
    const ContextScope scope(plan->context->handle);
    auto place = [&](const auto& forward, const auto& adjoint, auto& forward_data, auto& adjoint_data) {
        forward_data = Executable::Impl(plan->context, forward->Constants()).metadata;
        forward_data->Upload();
        adjoint_data
            = AdjointExecutable(plan->context, adjoint->Constants(), plan->n_qubits, plan->parameter_map).metadata;
        adjoint_data->Upload();
    };
    place(plan->right_forward, plan->right_adjoint, plan->right_forward_data, plan->right_adjoint_data);
    if (!hermitian) {
        place(plan->left_forward, plan->left_adjoint, plan->left_forward_data, plan->left_adjoint_data);
    }
    // Upload immutable observable snapshots once, never during evaluate().
    for (const auto& ham : hams) {
        plan->observables.push_back(owner.PrepareObservable(ham));
    }
    if (!hermitian) {
        for (const auto& ham : hams_dagger) {
            plan->daggers.push_back(owner.PrepareObservable(ham));
        }
    }
    // Bound retained memory to one real state, not one per Hamiltonian. Cache
    // the most expensive eligible sum; single-term operators need no expansion.
    Observable::Impl* cached = nullptr;
    for (const auto& observable : plan->observables) {
        auto* candidate = observable->impl_.get();
        if (candidate->diagonal_terms && (!cached || candidate->diagonal_terms > cached->diagonal_terms)) {
            cached = candidate;
        }
    }
    if (cached) {
        cached->CacheDiagonal();
    }
    impl_ = std::move(plan);
}

GradientPlan::~GradientPlan() = default;

GradientPlan::Result GradientPlan::Evaluate(const State& right_state, const State& left_state,
                                            const std::vector<parameter::ParameterResolver>& parameters) const {
    const auto& plan = *impl_;
    right_state.impl_->Validate(*left_state.impl_);
    if (right_state.impl_->context != plan.context || right_state.impl_->n_qubits != plan.n_qubits) {
        throw std::invalid_argument("Gradient plan requires the same Ascend device and qubit count.");
    }
    // Validate the ENTIRE batch, including unused resolver values, before any
    // binding, allocation, callback, or device work. Missing no_grad inputs are
    // errors too, though those names need not occupy gradient output columns.
    for (const auto& resolver : parameters) {
        ValidateResolver(resolver);
        for (const auto& name : plan.required_parameters) {
            if (!resolver.Contains(name)) {
                throw std::invalid_argument("Gradient input is missing parameter '" + name + "'.");
            }
        }
    }
    Result result(parameters.size());
    if (parameters.empty() || plan.observables.empty()) {
        return result;
    }
    // Lock aliases once and distinct inputs without lock-order inversions. No
    // mutable scratch belongs to the plan: independent evaluate calls can run
    // concurrently, while each call sees a stable pair of untouched inputs.
    const PairLock lock(right_state.impl_->mutex, left_state.impl_->mutex);
    const ContextScope scope(plan.context->handle);
    const auto plane = right_state.impl_->plane;
    const auto bytes = right_state.impl_->state.Size();
    Buffer right(plan.context, bytes), costate(plan.context, bytes);
    std::unique_ptr<Buffer> left, shared;
    // With one Hermitian observable, the final forward state can itself be
    // reconstructed in reverse. Preserve it only if another sweep needs it.
    if (!plan.hermitian || plan.observables.size() > 1) {
        shared = std::make_unique<Buffer>(plan.context, bytes);
    }
    if (!plan.hermitian) {
        left = std::make_unique<Buffer>(plan.context, bytes);
    }
    const auto output_size = static_cast<uint32_t>(plan.parameter_map.size() + 1);
    Buffer partials(plan.context, AdjointPartialBytes(plan.context->cores, output_size));
    std::unique_ptr<Buffer> workspace;
    Stream stream(plan.context);  // drains before all evaluation scratch is freed
    auto copy = [&](void* destination, void* source) {
        Check(aclrtMemcpyAsync(destination, bytes, source, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream.handle),
              "gradient copy D2D");
    };
    for (size_t sample = 0; sample < parameters.size(); ++sample) {
        const auto& resolver = parameters[sample];
        Executable::Impl forward_right(plan.context, plan.right_forward->Bind(resolver), plan.right_forward_data.get());
        AdjointExecutable adjoint_right(plan.context, plan.right_adjoint->Bind(resolver), plan.n_qubits,
                                        plan.parameter_map, plan.right_adjoint_data.get());
        std::unique_ptr<Executable::Impl> forward_left;
        std::unique_ptr<AdjointExecutable> adjoint_left;
        if (!plan.hermitian) {
            forward_left = std::make_unique<Executable::Impl>(plan.context, plan.left_forward->Bind(resolver),
                                                              plan.left_forward_data.get());
            adjoint_left = std::make_unique<AdjointExecutable>(plan.context, plan.left_adjoint->Bind(resolver),
                                                               plan.n_qubits, plan.parameter_map,
                                                               plan.left_adjoint_data.get());
        }
        const auto workspace_bytes = std::max({forward_right.workspace_bytes, adjoint_right.workspace_bytes,
                                               forward_left ? forward_left->workspace_bytes : 0U,
                                               adjoint_left ? adjoint_left->workspace_bytes : 0U});
        if (workspace_bytes && (!workspace || workspace->Size() < workspace_bytes)) {
            workspace = std::make_unique<Buffer>(plan.context, workspace_bytes);
        }
        Transaction transaction(stream.handle);  // before bound metadata can unwind
        forward_right.metadata->Upload(stream.handle);
        adjoint_right.metadata->Upload(stream.handle);
        if (forward_left) {
            forward_left->metadata->Upload(stream.handle);
            adjoint_left->metadata->Upload(stream.handle);
        }
        copy(right.Data(), right_state.impl_->state.Data());
        forward_right.Launch(stream.handle, right.Data(), plane, workspace ? workspace->Data() : nullptr);
        if (forward_left) {
            copy(left->Data(), left_state.impl_->state.Data());
            forward_left->Launch(stream.handle, left->Data(), plane, workspace ? workspace->Data() : nullptr);
        }
        result[sample].reserve(plan.observables.size());
        for (size_t h = 0; h < plan.observables.size(); ++h) {
            // Two/three (Hermitian) or four (general) scratch states. At most
            // one D2D shared-state copy per sweep, never per gate/derivative.
            auto* left_data = plan.hermitian ? right.Data() : left->Data();
            auto* sweep_state = shared ? shared->Data() : right.Data();
            if (shared) {
                copy(sweep_state, left_data);
            }
            plan.observables[h]->impl_->Launch(stream.handle, right.Data(), nullptr, costate.Data(), nullptr);
            const auto& left_sweep = plan.hermitian ? adjoint_right : *adjoint_left;
            left_sweep.Launch(stream.handle, sweep_state, costate.Data(), partials,
                              workspace ? workspace->Data() : nullptr);
            Check(aclrtSynchronizeStream(stream.handle), "left adjoint sweep");
            auto row = left_sweep.Read(partials);
            if (plan.hermitian) {
                for (size_t column = 1; column < row.size(); ++column) {
                    row[column] = {2 * row[column].real(), 0};
                }
            } else {
                copy(sweep_state, right.Data());
                plan.daggers[h]->impl_->Launch(stream.handle, left_data, nullptr, costate.Data(), nullptr);
                adjoint_right.Launch(stream.handle, sweep_state, costate.Data(), partials,
                                     workspace ? workspace->Data() : nullptr);
                Check(aclrtSynchronizeStream(stream.handle), "right adjoint sweep");
                const auto right_row = adjoint_right.Read(partials);
                for (size_t column = 1; column < row.size(); ++column) {
                    row[column] += std::conj(right_row[column]);
                }
            }
            // Keep the full complex overlap, including Hermitian FP32 diagnostics.
            result[sample].push_back(std::move(row));
        }
        transaction.Finish("evaluate gradient sample");
    }
    return result;
}
}  // namespace mindquantum::sim::ascend
