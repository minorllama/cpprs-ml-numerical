#pragma once
#include "worker_demo.hpp"

/// Own one reusable pipeline slot and its three stream dependency events.
struct Slot {
    PinnedArray input, output;
    DeviceArray device_in, device_out;
    Event uploaded, computed, downloaded;
    int chunk = -1;
    /// Allocate matching pinned and device buffers for one chunk.
    explicit Slot(std::size_t count) : input(count), output(count), device_in(count), device_out(count) {}
};

using Slots = std::vector<std::unique_ptr<Slot>>;

/// Wait for the download before reading or reusing the pinned slot.
inline void consume(Slot& slot, const std::vector<float>& table) {
    if (slot.chunk < 0) return;
    CUDA_CHECK(cudaEventSynchronize(slot.downloaded.value));
    verify_chunk(slot.output.data, slot.output.count, slot.chunk, table);
    slot.chunk = -1;
}

/// Fill the next chunk and queue its upload on the implicit default stream.
inline void upload(Slot& slot, int chunk, const std::vector<float>& table) {
    consume(slot, table);
    for (std::size_t i = 0; i < slot.input.count; ++i) slot.input.data[i] = input_value(i, chunk);
    CUDA_CHECK(cudaMemcpyAsync(slot.device_in.data, slot.input.data,
        slot.input.count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(slot.uploaded.value));
    slot.chunk = chunk;
}

/// Wait for input readiness, process the chunk on stream one, and record completion.
inline void compute(Slot& slot, cudaStream_t stream, int rounds) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, slot.uploaded.value, 0));
    const auto count = slot.device_in.count;
    const unsigned blocks = static_cast<unsigned>((count + 255) / 256);
    chunk_kernel<<<blocks, 256, 0, stream>>>(slot.device_in.data, slot.device_out.data, count, rounds);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(slot.computed.value, stream));
}

/// Wait for computation before copying its output on the implicit default stream.
inline void download(Slot& slot) {
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, slot.computed.value, 0));
    CUDA_CHECK(cudaMemcpyAsync(slot.output.data, slot.device_out.data,
        slot.output.count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(slot.downloaded.value));
}

/// Run compute n while the default stream downloads n-1 and uploads n+1.
inline void pipeline_steps(Slots& slots, cudaStream_t stream, const Options& opt,
                           const std::vector<float>& table) {
    upload(*slots[0], 0, table);
    for (int n = 0; n < opt.chunks; ++n) {
        compute(*slots[n % 3], stream, opt.rounds);
        if (n > 0) download(*slots[(n - 1) % 3]);
        if (n + 1 < opt.chunks) upload(*slots[(n + 1) % 3], n + 1, table);
    }
    download(*slots[(opt.chunks - 1) % 3]);
    for (auto& slot : slots) consume(*slot, table);
}

/// Execute the same chunk workload without inter-chunk overlap for comparison.
inline void serial_steps(Slot& slot, cudaStream_t stream, const Options& opt,
                         const std::vector<float>& table) {
    for (int n = 0; n < opt.chunks; ++n) {
        upload(slot, n, table);
        compute(slot, stream, opt.rounds);
        download(slot);
        consume(slot, table);
    }
}

/// Time the bounded three-slot pipeline, including host generation and verification.
inline void run_pipeline(const Options& opt) {
    Slots slots;
    for (int i = 0; i < 3; ++i) slots.push_back(std::make_unique<Slot>(opt.elements));
    Stream stream;
    const auto table = reference_table(opt.rounds);
    Drain drain;
    chunk_kernel<<<1, 256>>>(nullptr, nullptr, 0, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = Clock::now();
    if (opt.serial) serial_steps(*slots[0], stream.value, opt, table);
    else pipeline_steps(slots, stream.value, opt, table);
    finish_timing(start);
    std::cout << "verified " << opt.chunks << " chunks; slots=3; "
              << (opt.serial ? "serialized" : "pipelined") << '\n';
}
