/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_

#include <memory>
#include <deque>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <condition_variable>
#include <utility>
#include <mutex>
#include <cuda_runtime.h>
#include <functional>
#include <fstream>

#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
// #include "tensorflow/core/framework/tensor_buffer_hash.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"

namespace stream_executor {
class DeviceMemoryBase;
class Event;
class Stream;
} // namespace stream_executor

namespace tensorflow {

class TensorParams;
class TensorBuffer;
class Device;
class DeviceContext;
// class HashBuffer;

// A GPU memory allocator that implements a 'best-fit with coalescing'
// algorithm.
class GPUBFCAllocator : public BFCAllocator {
 public:
  // 'cuda_gpu_id' refers to the ID of the GPU device within
  // the process and must reference a valid ID in the process.
  GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                  const string& name);
  GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                  const GPUOptions& gpu_options, const string& name);
  virtual ~GPUBFCAllocator();

  void* AllocateRaw(size_t alignment, size_t num_bytes);
  void SwapOutOne();
  // void GetOrCreateHashBuffer(const Tensor* tensor, const string& tensor_name, HashBuffer** hash_buf) override;
  void AddEventToStream(const string& op_name,se::Stream* stream) override;

  void RecordSwapContext(const TensorParams& params, TensorBuffer* tensor_buf) override;

  void RecordTensorAccess(const string& tensor_name, TensorBuffer* tensor_buf, const uint64 _time) override;

  void Notify(TensorBuffer* tensor_buf) override;

  typedef std::function<void()> Closure;
  typedef std::function<void(Closure)> Runner;
  void CheckInput(const string& tensor_name, TensorBuffer* tensor_buf, bool* flag, bool before) override;
  void CheckInput(const string& tensor_name,
                  TensorBuffer* tensor_buf,
                  se::Event** e,
                  bool before,
                  Runner runner,
                  se::Stream* s) override;

static uint64_t iter;
 private:

 bool init = false;
  // store the tensor's swapping status
  struct SwapStatus {
    bool in;      
    bool out;
    bool swap_out;
    bool swap_in;

    bool wait_out;
    bool wait_in;

    void set_swapout() {
      in = false;
      swap_out = true;
    }

    void set_out() {
      swap_out = false;
      out = true;
    }

    void set_swapin() {
      swap_in = true;
    }

    void set_in() {
      out = false;
      swap_in = false;
      in = true;
    }

    void unset_swapout() {
      in = true;
      swap_out = false;
    }

    void unset_swapin() {
      swap_in = false;
    }

    void set_waitout() { wait_out = true; }
    void set_waitin() { wait_in = true; }

    bool is_swapout() { return (!in&&swap_out); }
    bool is_out() { return out; }
    bool is_swapin() { return swap_in; }
    bool is_in() { return in; }
    bool is_waitout() { return wait_out; }
    bool is_waitin() { return wait_in; }
  };

  // enum SwapStatus {
  //   IN,
  //   OUT,
  //   SWAPPING_IN,
  //   SWAPPING_OUT
  // };


  inline void MergeChunks(BFCAllocator::ChunkHandle h1, BFCAllocator::ChunkHandle h2) {
    CHECK(h1 != kInvalidChunkHandle && h2 != kInvalidChunkHandle);
    BFCAllocator::Chunk* c1 = ChunkFromHandle(h1);
    BFCAllocator::Chunk* c2 = ChunkFromHandle(h2);
    //CHECK(!c2->in_use());

    BFCAllocator::ChunkHandle h3 = c2->next;
    c1->next = h3;
    CHECK(c2->prev == h1);
    if (h3 != kInvalidChunkHandle) {
      BFCAllocator::Chunk* c3 = ChunkFromHandle(h3);
      c3->prev = h1;
    }

    // Set new size
    c1->size += c2->size;

    DeleteChunk(h2);
  }

  inline void MergeBuffers(const void* ptr1, const void* ptr2) {
    BFCAllocator::ChunkHandle h1 = region_manager_.get_handle(ptr1);
    BFCAllocator::ChunkHandle h2 = region_manager_.get_handle(ptr2);
    CHECK(h1 != kInvalidChunkHandle && h2 != kInvalidChunkHandle);
    MergeChunks(h1, h2);
  }

  inline void SplitBuffer(const void* ptr, size_t num_bytes) {
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
    CHECK(h != kInvalidChunkHandle)
        << "Asked for SplitChunk of pointer we never allocated: " << ptr;
    BFCAllocator::Chunk* c = ChunkFromHandle(h);
    size_t rounded_bytes = RoundedBytes(num_bytes);
    CHECK(c->requested_size > num_bytes)
        << "Rounded bytes of the split number of bytes is largger than requested size of pointer: " << ptr;
    SplitChunkInUse(h, rounded_bytes);
  }

  void SplitChunkInUse(BFCAllocator::ChunkHandle h, size_t num_bytes) {
    ChunkHandle h_new_chunk = AllocateChunk();
    Chunk* c = ChunkFromHandle(h);
    CHECK(c->in_use() && (c->bin_num == kInvalidBinNum));

    // Create a new chunk starting num_bytes after c
    BFCAllocator::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
    new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
    region_manager_.set_handle(new_chunk->ptr, h_new_chunk);

    // Set the new sizes of the chunks.
    new_chunk->size = c->size - num_bytes;
    c->size = num_bytes;

    // The new chunk is not in use.
    new_chunk->allocation_id = -1;

    // Maintain the pointers.
    // c <-> c_neighbor becomes
    // c <-> new_chunk <-> c_neighbor
    BFCAllocator::ChunkHandle h_neighbor = c->next;
    new_chunk->prev = h;
    new_chunk->next = h_neighbor;
    c->next = h_new_chunk;
    if (h_neighbor != kInvalidChunkHandle) {
      Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
      c_neighbor->prev = h_new_chunk;
    }

    // Add the newly free chunk to the free bin.
    InsertFreeChunkIntoBin(h_new_chunk);
  }

  void SwapIn(const string& tensor_name);

  inline bool SwapOut(const string& tensor_name) {
    return SwapOut(tensor_name, 0);
  }

  bool SwapOut(const string& tensor_name, const int64 retain_size);

  /* inline void SwapOutNaive(const string& tensor_name) {
    SwapOutNaive(tensor_name, 0);
  } */

  // API using the naive cuda_runtime api
/*   void SwapOutNaive(const string& tensor_name, const int64 retain_size);
  void SwapInNaive(const string& tensor_name); */

  void LoadSwapPolicy();

  mutable std::mutex lock_;

  std::mutex mu_;
  
  typedef std::pair<std::shared_ptr<std::condition_variable>, std::shared_ptr<std::mutex> > condition_variable_and_mutex;  

  struct TensorSwapParams {
    string tensor_name;
    Device* device;
    DeviceContext* device_context;
    TensorBuffer* tensor_buffer = nullptr;  // TODO(px): move this to std::hash<TensorBuffer*>()(tensor_buffer), as tensor_buffer may not exist during the real running.
    // HashBuffer* hash_buffer;
    std::pair<void*, int64> swapped_cpu_buffer; // set if buffer swapped out
    std::pair<void*, int64> swapped_gpu_buffer; // set if buffer swapped out
    void* in_gpu_src;
    condition_variable_and_mutex cv_mu;
    // volatile int data_ready; // false if buffer swapped out
    SwapStatus data_ready;
    bool need_dealloc;
    // for two usage:
    // 1. when swap out a tensor, need to deallocate it after enqueuing computation (do once)
    // 2. when swap out a tensor, need to waitfor d2h stream after enqueuing cmoputation (do once)
    // bool need_deallocate;
    // when meeting swap-in tensor, need to make comp.stream wait for h2d stream (do once)
    se::Event* out_e;     // only init once
    se::Event* in_e;      // only init once
    bool valid = true;
    bool need_in_addr;
    // bool can_deallocate_after_swap_out;
    // bool then_deallocate;
    float out_fraction;
    bool ram_or_gpu = false; // true if in ram; false if in GPU's mem
    volatile bool if_done = false;//spin lock for the swap
  };

  struct TriggerInfo {
    string tensor_name;
    int access_count;
    int out_trigger_count;  // 0 if tensor will not be swapped out
    int in_trigger_count;   // 0 if tensor is not a trigger node of any swap tensor
    int total_access_count; // total access count of this tensor in one step
    string in_tensor; // in_tensor will be swapped in if the tensor is accessed in_trigger_count times,
                      // do nothing if in_trigger equals 0
    string in_trigger;
    TensorSwapParams* out_params;  // swap params of in_tensor
    TensorSwapParams* in_params;   // swap params of this tensor
  };
  

  // std::unordered_map<size_t, HashBuffer*> hash_bufs_;
  // GPUBFCAllocator own the underlying HashBuffer
  // std::unordered_map<TensorBuffer*, HashBuffer*> hash_bufs_; // as the buf exist at pre-run time, so can use it as the key
  // std::unordered_map<std::string, HashBuffer*> hash_bufs_;


  std::unordered_map<std::string, TriggerInfo> swap_triggers_;

  std::unordered_map<std::string, TensorSwapParams> tensor_swap_params_map_;
  // using stream_executor::cuda::CUevent;
  std::unordered_map<std::string, CUevent> finish_event_map_;

  std::unordered_map<const TensorBuffer*, std::string> buffer_tensor_map_;

  // std::unordered_set<std::string> invalid_swap_;

  std::unordered_map<std::string, std::vector<uint64> > tensor_access_times_ GUARDED_BY(lock_);

  // static cudaStream_t device_to_device_stream_;

  // static cudaStream_t host_to_device_stream_;

  // static cudaStream_t device_to_host_stream_;

  // static cudaEvent_t cuda_event_;

  /* static void CUDART_CB CudaCallback(cudaStream_t stream, cudaError_t status, void* done) {
    auto func = static_cast<std::function<void()>* >(done);
    (*func)();
    delete func;
  } */
 bool LRUAccess(std::string tensor_name);
 void LRUAdd(std::string tensor_name,TensorSwapParams* param);
std::list<TensorSwapParams*> q;//the pointer to the swap info
std::unordered_map<std::string,  std::list<TensorSwapParams*>::iterator> table;//tensor name,the iter to the list
// std::set<void*> ifSwaped;//if this pointer belongs to someone that has been swapped out
 void InitTensorSwapInfo(const string& tensor_name);

  TF_DISALLOW_COPY_AND_ASSIGN(GPUBFCAllocator);
};

// Suballocator for GPU memory.
class GPUMemAllocator : public SubAllocator {
 public:
  // Note: stream_exec cannot be null.
  explicit GPUMemAllocator(se::StreamExecutor* stream_exec,
                           bool use_unified_memory)
      : stream_exec_(stream_exec), use_unified_memory_(use_unified_memory) {
    CHECK(stream_exec_ != nullptr);
  }
  ~GPUMemAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override {
    void* ptr = nullptr;
    if (num_bytes > 0) {
      if (use_unified_memory_) {
        ptr = stream_exec_->UnifiedMemoryAllocate(num_bytes);
      } else {
        ptr = stream_exec_->AllocateArray<char>(num_bytes).opaque();
      }
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (ptr != nullptr) {
      if (use_unified_memory_) {
        stream_exec_->UnifiedMemoryDeallocate(ptr);
      } else {
        se::DeviceMemoryBase gpu_ptr(ptr);
        stream_exec_->Deallocate(&gpu_ptr);
      }
    }
  }

 private:
  se::StreamExecutor* stream_exec_;  // not owned, non-null
  const bool use_unified_memory_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(GPUMemAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
