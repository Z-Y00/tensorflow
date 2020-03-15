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

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include <fstream>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>

// using perftools::gputools::DeviceMemoryBase;
// using perftools::gputools::Stream;

// #define _DEBUG
// #define _DEBUGV2

/* #define cudaCheckError(cudaCall) {                                                  \
    cudaError_t err = cudaCall;                                                       \
    if(err!=cudaSuccess) {                                                            \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
      exit(0);                                                                        \
    }                                                                                 \
} */

namespace tensorflow {

std::string GetEnv(const std::string& env_name) {
  const char* env_p = std::getenv(env_name.c_str());
  if (env_p == nullptr) return "";
  return env_p;
}

const std::string swap_policy_env = "SWAP_POLICY_FILE";

const int64 kCopyThreshold = 10 << 20;    // 100M

/* cudaStream_t GPUBFCAllocator::device_to_device_stream_;
cudaStream_t GPUBFCAllocator::host_to_device_stream_;
cudaStream_t GPUBFCAllocator::device_to_host_stream_;
cudaEvent_t GPUBFCAllocator::cuda_event_; */
// static
 uint64_t GPUBFCAllocator::iter=0;

GPUBFCAllocator::GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                                 const string& name)
    : GPUBFCAllocator(cuda_gpu_id, total_memory, GPUOptions(), name) {}

GPUBFCAllocator::GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                                 const GPUOptions& gpu_options,
                                 const string& name)
    : BFCAllocator(
          new GPUMemAllocator(
              GpuIdUtil::ExecutorForCudaGpuId(cuda_gpu_id).ValueOrDie(),
              gpu_options.per_process_gpu_memory_fraction() > 1.0 ||
                  gpu_options.experimental().use_unified_memory()),
          total_memory, gpu_options.allow_growth(), name) {
      // LoadSwapPolicy();
     /*  cudaStreamCreate(&device_to_device_stream_);
      cudaStreamCreate(&host_to_device_stream_);
      cudaStreamCreate(&device_to_host_stream_);
      cudaEventCreate(&cuda_event_); */
    }

GPUBFCAllocator::~GPUBFCAllocator() {
  /* const std::string invalid_swap_filename = "/tmp/invalid_swap.txt";
  std::fstream fout(invalid_swap_filename, fout.out);
  if (!fout.is_open()) {
    LOG(ERROR) << "Fail to open invalid swap file";
    return;
  }
  for (auto& name : invalid_swap_) {
    fout << name << "\n";
  } */
}

/*----------------to be deprecated---------------*/
/* void GPUBFCAllocator::GetOrCreateHashBuffer(const Tensor* tensor, const string& tensor_name, HashBuffer** hash_buf){
  // Only record the necessary tensor
  if (tensor_swap_params_map_.count(tensor_name) == 0) return;
  std::lock_guard<std::mutex> l(lock_);

  auto t_buf = tensor->buf_;
  // auto t_name = tensor->Name();  // this field can be empty
  if (t_buf == nullptr) {
    LOG(FATAL) << "Buffer should not be null!";
    return;
  }

  if (hash_bufs_.count(tensor_name) == 0) {
    *hash_buf = new HashBuffer(t_buf, tensor_name);
    hash_bufs_[tensor_name] = *hash_buf;
    return;
  } else {
    *hash_buf = hash_bufs_[tensor_name];
    return;
  }
} */

#include<iostream>
// A linked list + hashtable of pointers to the linked list nodes is the usual way to implement LRU caches. This gives O(1) operations (assuming a decent hash). Advantage of this (being O(1)): you can do a multithreaded version by just locking the whole structure. You don't have to worry about granular locking etc.
// 
// Briefly, the way it works:
// 
// On an access of a value, you move the corresponding node in the linked list to the head.
// 
// When you need to remove a value from the cache, you remove from the tail end.
// 
// When you add a value to cache, you just place it at the head of the linked list. 
void GPUBFCAllocator::LRUAdd(std::string tensor_name,TensorSwapParams* param){
    // std::cerr<<"ad";
    q.push_front(param);
    table[tensor_name]=q.begin();//add to the front
    // std::cerr<<"d";
}
bool GPUBFCAllocator::LRUAccess(std::string tensor_name){
    // std::cerr<<"ac";
    std::list<TensorSwapParams*>::iterator pointer = table[tensor_name];
    q.push_front(*pointer);
    q.erase(pointer);//erase the iter
    table[tensor_name]=q.begin();//move to front
    // std::cerr<<"c";
    return true;// if the tensor exit
}
void GPUBFCAllocator::ResetLRU(){
  for (auto& param : q){
    if(param!=nullptr){
      param->ram_or_gpu = false;//in GPU
      param->if_done = true;
      param->if_done_in = true;
    }else{
      //????
      LOG(INFO)<<"WTF??";
    }
  }
}

// #define Baseline
#ifndef Baseline
void* GPUBFCAllocator::AllocateRaw(size_t unused_alignment, size_t num_bytes) {
  if((num_bytes>>20)>120)
    LOG(INFO)<<"Alloc: "<<(num_bytes>>20);
  void* ans = BFCAllocator::AllocateRawInternal(unused_alignment,num_bytes, false);
  if(ans!=nullptr) return ans;

  // num_bytes += 0x100;

  if(q.size()==0){
      LOG(FATAL)<<"q empty";
      exit(1);
  }
  std::list<TensorSwapParams*>::iterator _iter = --q.end();
  while(ans==nullptr){
    if(_iter==q.begin()){
      LOG(FATAL)<<"full for trying in M: "<<(num_bytes>>20);
      exit(1);
    }
    if(!(*_iter)->ram_or_gpu){//if in GPU
      //  void* src_ptr = (void*)(iter->tensor_buffer->data());//get curr chunk size
      //  avail += RequestedSize(src_ptr);
      //  LOG(INFO)<<"Try SwapOut "<<(*_iter)->tensor_name;//<<RequestedSize((void*)((*iter)->tensor_buffer->data()));
       if(!SwapOut((*_iter)->tensor_name)){
        _iter--;
        continue;
       }
      //  while(!(*_iter)->if_done)
          // ;//spin lock
       
       ans = BFCAllocator::AllocateRawInternal(unused_alignment,num_bytes, false);
        if(ans!=nullptr){
          break;
       }
      //  LOG(INFO)<<"failed";
    }else{
      LOG(INFO)<<"In Ram "<<(*_iter)->tensor_name;
    }
    _iter--; //get the one before it

    // LOG(INFO)<<"try again";
  }
  // ans+=0x100;
  // LOG(INFO)<<"Success! ";//<<ans;//if the output is +1 then, it is tf's internal pointer
  return ans;
}
#endif
static int record_step = 9;
static int saved_iter = 0;

void GPUBFCAllocator::InitTensorSwapInfo(const string& tensor_name){
    if(saved_iter!=record_step){//recording
      return;
    }
   if (tensor_swap_params_map_.count(tensor_name) != 0) 
     return;
        // std::cerr<<"i ";

        // LOG(INFO)<<iter<<" init "<<tensor_name;
        //init the struct
        //init the CUevent without timing
        auto& event = finish_event_map_[tensor_name];//access to create
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

        auto& swap_params = tensor_swap_params_map_[tensor_name];
        swap_params.tensor_name = tensor_name;
        swap_params.cv_mu = std::make_pair(std::make_shared<std::condition_variable>(), std::make_shared<std::mutex>());

        //no trigger !
        // auto& swap_out_trigger = swap_triggers_[out_tensor_name];
        // swap_out_trigger.tensor_name = out_tensor_name;
        // swap_out_trigger.out_trigger_count = out_trigger_count;
        // swap_out_trigger.out_params = &swap_params;
        // swap_out_trigger.in_trigger = tensor_name;
        // swap_out_trigger.access_count = 0;
        // swap_out_trigger.total_access_count = out_tensor_total_access;

        // auto& swap_in_trigger = swap_triggers_[tensor_name];
        // swap_in_trigger.tensor_name = tensor_name;
        // swap_in_trigger.in_trigger_count = in_trigger_count;
        // swap_in_trigger.in_tensor = out_tensor_name;
        // swap_in_trigger.in_params = &swap_params;
        // swap_in_trigger.access_count = 0;
        // swap_in_trigger.total_access_count = in_trigger_total_access;
}

void GPUBFCAllocator::RecordTensorAccess(const string& tensor_name,
                                         TensorBuffer* buf,
                                         const uint64 _time) {
  if(tensor_name.find("v/tower_0/gpu_cached_images")!=std::string::npos){ 
    //  LOG(INFO)<<tensor_name;
     return;
  } 
  if(saved_iter != iter){
    saved_iter = iter;
    ResetLRU();
    LOG(INFO)<<"reset LRU";
  }
  if(saved_iter<record_step){
    return;
  } else if(saved_iter==record_step){//recording
        // std::cerr<<"i";//<<std::endl;
        InitTensorSwapInfo(tensor_name);
        auto& swap_params = tensor_swap_params_map_[tensor_name];
        LRUAdd(tensor_name,&(swap_params));//add to the LRU
  }else{
      //  LOG(INFO)<<tensor_name;
      LRUAccess(tensor_name);
  }
    auto& swap_params = tensor_swap_params_map_[tensor_name];
    if(swap_params.tensor_buffer == nullptr){
     // LOG(INFO)<<"NULL "<<tensor_name;
     return;
    }else if(swap_params.tensor_buffer->data()==nullptr){//not in
     if(strcmp(tensor_name.c_str(),swap_params.tensor_buffer->GetMainName().c_str())!=0){
      LOG(INFO)<<"swapin shared "<<tensor_name<<" with main "<<swap_params.tensor_buffer->GetMainName();
      SwapIn(swap_params.tensor_buffer->GetMainName());
      auto& params = tensor_swap_params_map_[swap_params.tensor_buffer->GetMainName()];
      // LOG(INFO)<<"spin on "<<params.tensor_name<<" "<<(void*)&params.if_done_in;
      // while(!params.if_done_in)//default is true, which will break and not spin
        //  ;//spin lock on the mainName's bool
 
      // LOG(INFO)<<"nice";
      if(swap_params.tensor_buffer->data()==nullptr){
          LOG(FATAL)<<"Sad";
      }
      return;
    }
    }else{//already good
      return;
    }


    SwapIn(tensor_name);//force swapin what to be access
      // when a tensor has been swapped out, we need to set the tensor_buffer->data() to the swapin corresponding memory addr
      DCHECK(tensor_swap_params_map_.count(tensor_name));
      auto& cv_mu = swap_params.cv_mu;
      {
        // std::lock_guard<std::mutex> ll(lock_);  // wait swapin finish if
        std::lock_guard<std::mutex> l(*(cv_mu.second));
        // TODO(px): can be replaced as ready.is_in() || ready.is_swapin()?
        if (swap_params.need_in_addr) {
          void* in_gpu_src = swap_params.in_gpu_src;
          if (in_gpu_src == nullptr) {
            LOG(FATAL) << "Weird!" << tensor_name << ": the correspoding SwapIn in_gpu_src is not set!";
          }
          buf->set_data(in_gpu_src);
          swap_params.need_in_addr = false;
        #ifdef _DEBUGV2
          LOG(INFO) << "Set " << tensor_name << " buffer addr";
        #endif
        }
      }
    // while(!swap_params.if_done_in)//default is true, which will break and not spin
        //  ;//spin lock
    return;

  if (tensor_swap_params_map_.count(tensor_name)) {
    SwapIn(tensor_name);
  }

  if (swap_triggers_.count(tensor_name) == 0) {
    return;
  }

  auto& swap_trigger = swap_triggers_[tensor_name];
  int cnt;
  {
    std::lock_guard<std::mutex> l(mu_);
    swap_trigger.access_count++;
    cnt = swap_trigger.access_count;
    if (swap_trigger.access_count == swap_trigger.total_access_count) {
      swap_trigger.access_count = 0;
    }
  }

  if (swap_trigger.out_trigger_count != 0) {
    if (cnt == swap_trigger.out_trigger_count) {
      SwapOut(tensor_name);
    } else if (cnt > swap_trigger.out_trigger_count) {
      // when a tensor has been swapped out, we need to set the tensor_buffer->data() to the swapin corresponding memory addr
      DCHECK(tensor_swap_params_map_.count(tensor_name));
      auto& swap_params = tensor_swap_params_map_[tensor_name];
      auto& cv_mu = swap_params.cv_mu;
      {
        // std::lock_guard<std::mutex> ll(lock_);  // wait swapin finish if
        std::lock_guard<std::mutex> l(*(cv_mu.second));
        // TODO(px): can be replaced as ready.is_in() || ready.is_swapin()?
        if (swap_params.need_in_addr) {
          void* in_gpu_src = swap_params.in_gpu_src;
          if (in_gpu_src == nullptr) {
            LOG(FATAL) << "Weird!" << tensor_name << ": the correspoding SwapIn in_gpu_src is not set!";
          }
          buf->set_data(in_gpu_src);
          swap_params.need_in_addr = false;
        #ifdef _DEBUGV2
          LOG(INFO) << "Set " << tensor_name << " buffer addr";
        #endif
        }
      }
    }
  }

  if (swap_trigger.in_trigger_count != 0 && cnt == swap_trigger.in_trigger_count) {
    SwapIn(swap_trigger.in_tensor);
  }
}

void GPUBFCAllocator::AddEventToStream(const string& op_name,se::Stream* stream){
  return;
  if(finish_event_map_.count(op_name)!=0){
    auto& event = finish_event_map_[op_name];// this will add the event to the stream
    CUresult res = cuEventRecord(event, stream_executor::cuda::AsCUDAStream(stream)->cuda_stream());
    switch (res) {
      case CUDA_SUCCESS:
        return ;
      case CUDA_ERROR_DEINITIALIZED:
      case CUDA_ERROR_NOT_INITIALIZED:
      default:
    LOG(INFO) << "Weird! cuEventRecord call failed";
    }
  }else{
    LOG(INFO) << "AddEventToStream not found: "<<op_name;
  }
}


void GPUBFCAllocator::RecordSwapContext(const TensorParams& params, TensorBuffer* tensor_buf) {
  if(params.name.find("v/tower_0/gpu_cached_images")!=std::string::npos){ 
       LOG(INFO)<<"skip tensor";
       return;
     } 
  InitTensorSwapInfo(params.name);
  if (tensor_swap_params_map_.count(params.name) == 0) return;
  if(tensor_buf==nullptr) return;
  // LOG(INFO)<<"RecordSwapContext";
  std::lock_guard<std::mutex> l(lock_);
  const string &tensor_name = params.name;
  TensorSwapParams& swap_params = tensor_swap_params_map_[tensor_name];
  swap_params.device = params.device; 
  swap_params.device_context = params.device_context;
  swap_params.tensor_buffer = tensor_buf;
  swap_params.in_gpu_src = nullptr;
  LOG(INFO)<<"SwapContext "<<params.name;
  tensor_buf->AddCrossRef((void**)&swap_params.tensor_buffer,tensor_name);
  // if(strcmp("v/tower_0/gpu_cached_images/read:0",tensor_name.c_str())==0){ 
    // tensor_buf->Trace();
    // LOG(INFO)<<tensor_name;
    // LOG(INFO)<<tensor_buf->xRef;
    // LOG(INFO)<< tensor_name<<tensor_buf<<" "<<(void**)&swap_params.tensor_buffer<<" "<<(void*)(tensor_buf->data());
  // } 
  // if (hash_bufs_.count(tensor_name) == 0) {
  // #ifdef _DEBUGV2
    // LOG(INFO) << "New HashBuffer for " << tensor_name<<tensor_buf->UsingCount();
  // #endif
  //   hash_bufs_[tensor_name] = new HashBuffer(tensor_buf, tensor_name);
  // }
  // swap_params.hash_buffer = hash_bufs_[tensor_name];
  // swap_params.data_ready = SwapStatus::IN;
  swap_params.data_ready = {true, false, false, false, false, false};
  swap_params.need_dealloc = false;
  // swap_params.need_deallocate = false;
  swap_params.need_in_addr = false;
  // swap_params.can_deallocate_after_swap_out = true;
  // swap_params.then_deallocate = false;
  buffer_tensor_map_[tensor_buf] = tensor_name;
}

// TODO(px): no need for now
void GPUBFCAllocator::Notify(TensorBuffer* tensor_buffer) {
  return;
  /* if (buffer_tensor_map_.count(tensor_buffer) == 0) return;
  const string& tensor_name = buffer_tensor_map_[tensor_buffer];
  auto& swap_params = tensor_swap_params_map_[tensor_name];
  auto& cv_mu = swap_params.cv_mu;
  std::lock_guard<std::mutex> l(*(cv_mu.second));
  int64 gpu_part_size = swap_params.swapped_gpu_buffer.second;
  if (swap_params.can_deallocate_after_swap_out && swap_params.then_deallocate) {
    if (gpu_part_size <= 0)
      DeallocateRaw(tensor_buffer->data());
    else
      SplitBuffer(tensor_buffer->data(), gpu_part_size);
    tensor_buffer->set_data(nullptr);
    swap_params.data_ready = SwapStatus::OUT;
    swap_params.then_deallocate = false;
  }

  // this tensor is left compuation to deallocate it and it's a useless swap which means
  // it's no need to deallocate it after swapping out, so we set the SwapStatus to IN
  // directly, and leave the computation to set the status instead of SwapIn is delay the
  // computation to alleviate the memory pressure, but may lose some performance.
  if (!swap_params.can_deallocate_after_swap_out && swap_params.then_deallocate) {
    LOG(INFO) << "Set status IN for " << swap_params.tensor_name;
    swap_params.data_ready = SwapStatus::IN;
    cv_mu.first->notify_all();
  } */
}

// void TryReleaseBuffer(const string& tensor_name, TensorBuffer* tensor_buf) {
//   if (tensor_swap_params_map_.count(tensor_name) == 0) return;

//   auto& swap_params = tensor_swap_params_map_[tensor_name];
//   auto& cv_mu = swap_params.cv_mu;
//   {
//     std::lock_guard<std::mutex> l(*(cv_mu.second));
//     if (swap_params.need_deallocate) {
//       DeallocateRaw(tensor_buf->data());
//       swap_params.need_deallocate = false;
//     }
//   }
// }

// Check inputs again when enqueuing computation
// 1. see if need to deallocate swapped-out tensor's memory and wait for device_to_host_stream (just the computation which trigger swapped-out)
// 2. see if need to wait for host_to_device_stream (the computation which need swapped-out tensor) (set tensor_buf->data() need to be before the enqueuing of computation)
void GPUBFCAllocator::CheckInput(const string& tensor_name,
                                  TensorBuffer* tensor_buf,
                                  bool* flag,
                                  bool before) {
  InitTensorSwapInfo(tensor_name);
  if (tensor_swap_params_map_.count(tensor_name) == 0) return;

  auto& swap_params = tensor_swap_params_map_[tensor_name];
  auto& cv_mu = swap_params.cv_mu;
  {
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    auto& ready = swap_params.data_ready;
    if (before) {
      // check tensor iff swapping-in before comp
      if (ready.is_swapin()) {
        *flag = true;
      #ifdef _DEBUGV2
        LOG(INFO) << tensor_name << " : wait h2d to true";
      #endif
      } else if (ready.is_in()) {
      #ifdef _DEBUGV2
        LOG(INFO) << tensor_name << " is already been swapped-in, dont wait h2d";
      #endif
      }
    } else {
      // check tensor iff swapped-out after comp
      if (ready.is_swapout()) {
        DeallocateRaw(tensor_buf->data());
        *flag = true;
      #ifdef _DEBUGV2
        LOG(INFO) << tensor_name << " : wait d2h to true";
      #endif
      } else if (ready.is_out()) {
        DeallocateRaw(tensor_buf->data());
      #ifdef _DEBUGV2
        LOG(INFO) << tensor_name << " is already been swapped-out, dont wait d2h";
      #endif
      }
    }
  }
}

void GPUBFCAllocator::CheckInput(const string& tensor_name,
                                 TensorBuffer* tensor_buf,
                                 se::Event** e,
                                 bool before,
                                 Runner runner,
                                 se::Stream* s) {

  InitTensorSwapInfo(tensor_name);

  if (tensor_swap_params_map_.count(tensor_name) == 0) return;

  auto& swap_params = tensor_swap_params_map_[tensor_name];
  auto& cv_mu = swap_params.cv_mu;
  {
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    // int ready = swap_params.data_ready;
    auto& ready = swap_params.data_ready;
    if (before) {
      // (px): play double check to avoid waitforevent, maybe not necessary
      // the ready can be OUT as when swapout is done, it will overwrite the
      // swapstatus after swapin
      if (ready.is_swapin()) {
        if (ready.is_waitin()) {
          // TODO(px): as this swap-in check happen before enqueuing the computation into stream,
          // how we can make sure that the wait one will be enqueued into stream earlier?
          return;
        }
        if (swap_params.in_e == nullptr) {
          LOG(FATAL) << tensor_name << " swap in event is nullptr!";
        }
        *e = swap_params.in_e;
      #ifdef _DEBUGV2
        LOG(INFO) << "Wait " << tensor_name << " swap in event.";
      #endif
        // swap_params.need_wait_in = false;
        swap_params.data_ready.set_waitin();
        auto done = [&swap_params] () {
          auto& cv_mu = swap_params.cv_mu;
          std::lock_guard<std::mutex> l(*(cv_mu.second));
          // LOG(INFO) << "Check " << swap_params.tensor_name << " swap in status";
          auto& ready = swap_params.data_ready;
          // CHECK(ready.is_in() || ready.is_swapin());
          if (!(ready.is_in() || ready.is_swapin())) {
            LOG(FATAL) << swap_params.tensor_name << " status: " << (ready.is_out() ? 1 : 0);
          }
          if (ready.is_swapin()) {
            LOG(INFO) << swap_params.tensor_name << " not finish swap in before comp.";
          }
        };
        runner(done);
      }
    } else {
      if (ready.is_swapout()) {
        if (ready.is_waitout()) {
          return;
        }
        if (swap_params.out_e == nullptr) {
          LOG(FATAL) << tensor_name << " swap out event is nullptr!";
        }
        if (swap_params.need_dealloc) {
        #ifdef _DEBUG2
          LOG(INFO) << "Deallocate " << swap_params.tensor_name << " when enqueue comp success.";
        #endif
          auto wait_out_event = [=] () {//swapout wait 
            s->ThenWaitFor(swap_params.out_e);
          };
          DeallocateRawSwap(tensor_buf->data(), wait_out_event);//
          swap_params.need_dealloc = false;
        }
        // *e = swap_params.out_e;
        // swap_params.need_wait_out = false;
        swap_params.data_ready.set_waitout();
        swap_params.ram_or_gpu = false;
        swap_params.if_done = true;
      #ifdef _DEBUGV2
        LOG(INFO) << "Wait " << tensor_name << " swap out event.";
      #endif
        // auto done = [&swap_params] () {
        //   auto& cv_mu = swap_params.cv_mu;
        //   std::lock_guard<std::mutex> l(*(cv_mu.second));
        //   // LOG(INFO) << "Check " << swap_params.tensor_name << " swap out status";
        //   auto& ready = swap_params.data_ready;
        //   // CHECK(ready.is_out() || ready.is_swapout());
        //   if (!(ready.is_out() || ready.is_swapout())) {
        //     LOG(FATAL) << swap_params.tensor_name << " status: " << (ready.is_in() ? 1 : 0);
        //   }
        //   if (ready.is_swapout()) {
        //   #ifdef _DEBUG
        //     LOG(INFO) << swap_params.tensor_name << " not finish swap out when comp finish.";
        //   #endif
        //   }          
        // };
        // runner(done);
      }
    }
  }
}

void GPUBFCAllocator::LoadSwapPolicy() {
  std::string swap_policy_file = GetEnv(swap_policy_env);
  if (swap_policy_file.empty()) {
  #ifdef _DEBUG
    LOG(INFO) << "No swap policy specified";
  #endif
    return;
  }
  std::fstream fin(swap_policy_file, fin.in);
  if (!fin.is_open()) {
  #ifdef _DEBUG
    LOG(INFO) << "open " << swap_policy_file << " failed.";
  #endif
    return;
  }

  LOG(INFO) << "Load " << swap_policy_file << " succeeded.";

  string out_tensor_name, in_trigger_name;
  int out_trigger_count, in_trigger_count;
  int out_tensor_total_access, in_trigger_total_access;
  while(fin >> out_tensor_name >> out_tensor_total_access >> out_trigger_count
            >> in_trigger_name >> in_trigger_total_access >> in_trigger_count) {
    if (out_tensor_name[0] == '#') {
      continue;
    }
    //init the CUevent without timing
    auto& event = finish_event_map_[out_tensor_name];//access to create
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

    auto& swap_params = tensor_swap_params_map_[out_tensor_name];
    swap_params.tensor_name = out_tensor_name;
    swap_params.cv_mu = std::make_pair(std::make_shared<std::condition_variable>(), std::make_shared<std::mutex>());

    auto& swap_out_trigger = swap_triggers_[out_tensor_name];
    swap_out_trigger.tensor_name = out_tensor_name;
    swap_out_trigger.out_trigger_count = out_trigger_count;
    swap_out_trigger.out_params = &swap_params;
    swap_out_trigger.in_trigger = in_trigger_name;
    swap_out_trigger.access_count = 0;
    swap_out_trigger.total_access_count = out_tensor_total_access;

    auto& swap_in_trigger = swap_triggers_[in_trigger_name];
    swap_in_trigger.tensor_name = in_trigger_name;
    swap_in_trigger.in_trigger_count = in_trigger_count;
    swap_in_trigger.in_tensor = out_tensor_name;
    swap_in_trigger.in_params = &swap_params;
    swap_in_trigger.access_count = 0;
    swap_in_trigger.total_access_count = in_trigger_total_access;
  }
  fin.close();
}

Status PrepareCopy(Device* device, const DeviceContext* ctx,
    const DeviceBase::GpuDeviceInfo** dev_info, se::Stream** stream) {
  if (device == nullptr) {
    return errors::Internal("Unexpected null device.");
  }
  auto di = device->tensorflow_gpu_device_info();
  if (di == nullptr) {
    return errors::Internal("Unexpected null device info.");
  }
  *dev_info = di;
  if (ctx == nullptr) {
    return errors::Internal("Unexpected null device context.");
  }
  auto gs = static_cast<const GPUDeviceContext*>(ctx)->stream();
  if (gs == nullptr) {
    return errors::Internal("No gpu stream is available.");
  }
  *stream = gs;
  return Status::OK();
}

/*----------------current implem SwapOut---------------*/

bool GPUBFCAllocator::SwapOut(const string& tensor_name, const int64 retain_size) {
  /* if (invalid_swap_.count(tensor_name) != 0){
  #ifdef _DEBUGV2
    LOG(INFO) << "Ignore the invalid swap out: " << tensor_name;
  #endif
    return;
  } */

  if (tensor_swap_params_map_.count(tensor_name) == 0) {
    LOG(INFO) <<"not init: "<< tensor_name;
    return false;
  }
  
  DCHECK(tensor_swap_params_map_.count(tensor_name));
  auto &swap_params = tensor_swap_params_map_[tensor_name];

  auto &cv_mu = swap_params.cv_mu;
  std::lock_guard<std::mutex> l(*(cv_mu.second));
  if(swap_params.ram_or_gpu){//if in RAM
    LOG(INFO)<<"in ram "<<tensor_name;
    return false;
  }
  swap_params.data_ready.set_swapout();
  swap_params.if_done = false;


  // HashBuffer* hash_buffer = swap_params.hash_buffer;
  // LOG(INFO)<<tensor_buffer->UsingCount();// all 0??????
  TensorBuffer* tensor_buffer = swap_params.tensor_buffer;
  if(tensor_buffer==nullptr){//some were not recorded?
    swap_params.data_ready.unset_swapout();
    LOG(INFO)<<"buffer==nullptr "<< tensor_name;
    return false;
  }
  
  // if(strcmp("v/tower_0/L2Loss:0",tensor_name.c_str())==0){
    //  swap_params.if_done = true;
    //  return true;
    // LOG(INFO)<< tensor_name<<tensor_buffer<<" "<<(void*)&swap_params.tensor_buffer;
    // LOG(INFO)<< (void*)tensor_buffer->data();
  // }
  // std::cerr<< tensor_name<<" ";

  // if(tensor_buffer->UsingCount() == 0){ }
  // std::cerr<<"i ";

  void* src_ptr = (void*)(tensor_buffer->data());
  //interesting??? we swapped out one that has been deconstructed
  if(src_ptr==nullptr){//means this buffer is shared, and other has swapped out it
    LOG(INFO)<<"already swappout, giveup "<< tensor_name;
    swap_params.ram_or_gpu = true;
    swap_params.data_ready.set_out();
    return false;
  }
  int64 total_bytes = RequestedSize(src_ptr);

  int64 gpu_part_size, cpu_part_size;
    gpu_part_size = 0;
    cpu_part_size = total_bytes;//this is always true!

  if (cpu_part_size < kCopyThreshold) {
  #ifdef _DEBUGV2
    LOG(INFO) << tensor_name << " memory size is below kCopyThreshold, ignore it!";
  #endif
    // std::lock_guard<std::mutex> l(*(cv_mu.second));
    // swap_params.data_ready = SwapStatus::IN;
    swap_params.data_ready.unset_swapout();
    swap_params.valid = false;
    return false;    
  }
    // LOG(INFO)<<tensor_name<<" size M "<<(total_bytes>>20);
#ifdef _DEBUG
  LOG(INFO) << "Start to swap out: " << tensor_name;
#endif


  Device* device = swap_params.device;
  DeviceContext* device_context = swap_params.device_context;
  const DeviceBase::GpuDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(device, device_context, &dev_info, &send_stream);
  if (!s.ok()) {
    LOG(FATAL) << "PrepareCopy failed.";
    // std::lock_guard<std::mutex> l(*(cv_mu.second));
    // swap_params.data_ready = SwapStatus::IN;
    swap_params.data_ready.unset_swapout();
    return false;
  }

#ifdef _DEBUGV2
  LOG(INFO) << "PrepareCopy success: " << tensor_name;
#endif

  swap_params.swapped_gpu_buffer = std::make_pair(src_ptr, gpu_part_size);

  static Allocator* cuda_host_allocator = GPUProcessState::singleton()->GetCUDAHostAllocator(0);
  void* cpu_part_dst_ptr = cuda_host_allocator->AllocateRaw(0, cpu_part_size);
  swap_params.swapped_cpu_buffer = std::make_pair(cpu_part_dst_ptr, cpu_part_size);

  if (cpu_part_dst_ptr == nullptr) {
    LOG(FATAL) << "Allocate host memory failed.";
    // std::lock_guard<std::mutex> l(*(cv_mu.second));
    // swap_params.data_ready = SwapStatus::IN;
    swap_params.data_ready.unset_swapout();
    return false;
  }


  auto device_to_host_stream =
      static_cast<const GPUDeviceContext*>(device_context)->device_to_host_stream();
  if (device_to_host_stream == nullptr) {
    LOG(FATAL) << "No device_to_host_stream is available.";
    // std::lock_guard<std::mutex> l(*(cv_mu.second));
    // swap_params.data_ready = SwapStatus::IN;
    swap_params.data_ready.unset_swapout();
    return false;
  }
  // Wait for the sender's main stream to make sure the data are available.
  // px: As the kernel in compute() function in ExecutorState::Process() is async,
  // we need to wait the stream to complete current computation
  // TODO(px): this ThenWaitFor is a coarse-grain sync, as the tensor's producer kernel may finish and queue a lot of other kernel
  // and still we need to wait for these useless kernel to finish so we can execute our swapping func properly.
  // Maybe we can add a event to indicate the completion of tensor's producer kernel and wait only for that particular event.
  // No need to record this time as the ThenWaitFor will not block the host
  ///rgy think about this to event, maybe add finish event? send_stream, compute stream
  //this event need to be accessed at the tensor's execute and here
  // if(finish_event_map_.count(tensor_name)==0){
  device_to_host_stream->ThenWaitFor(send_stream);
  // }else{
  //   CUresult res = cuStreamWaitEvent(
  //     stream_executor::cuda::AsCUDAStream(device_to_host_stream)->cuda_stream(),
  //     finish_event_map_[tensor_name],
  //     0
  //     );
  //   // LOG(INFO)<<"hacked!";
  //     if (res != CUDA_SUCCESS) {
  //       LOG(ERROR) << "could not wait stream on event: ";
  //       //  << ToString(res);
  //       // return false;
  //   }
  // }


  se::DeviceMemoryBase gpu_src_ptr((void*)((uintptr_t)src_ptr + gpu_part_size), cpu_part_size);
  device_to_host_stream->ThenMemcpy(cpu_part_dst_ptr, gpu_src_ptr, cpu_part_size);
  // TODO(px): add a event after each swapping, get a event to indicate whether swapping finish, and this event should not be used for other use
    // std::lock_guard<std::mutex> l(*(cv_mu.second));
  dev_info->event_mgr->ThenRecordEvent(device_to_host_stream, &swap_params.out_e);
  // swap_params.need_wait_out = true;
  // swap_params.need_dealloc = true;

//todo, free the chunk if alloced! 2020-3-10
  // Use of the input may outlive stack scope, so keep a ref.
  // (px): keep the ref may cause the GPU memory being exhausted.
  // And there is no need to keep this ref, it's enough that knowing the swapping address.
  // tensor_buffer->Ref();
  swap_params.tensor_buffer->set_data(nullptr);

  dev_info->event_mgr->ThenExecute(
      device_to_host_stream,
      [this, device_to_host_stream, &swap_params, total_bytes]() {
        if (!device_to_host_stream->ok()) {
          LOG(FATAL) << "GPU->CPU Memcpy failed";
          std::lock_guard<std::mutex> l(*(swap_params.cv_mu.second));
          // swap_params.data_ready = SwapStatus::IN;
          swap_params.data_ready.unset_swapout();
          // tensor_buffer->Unref();
          return;
        }
        auto &cv_mu = swap_params.cv_mu;
        // std::unique_lock<std::mutex> lk(*(cv_mu.second));
        std::lock_guard<std::mutex> lk(*(cv_mu.second));
        // swap_params.data_ready = SwapStatus::OUT;
        swap_params.data_ready.set_out();
        // if (swap_params.need_dealloc) {
        #ifdef _DEBUGV2
          LOG(INFO) << "Deallocate " << swap_params.tensor_name << " when swapout done";
        #endif
          // DeallocateRaw(swap_params.swapped_gpu_buffer.first);
          // swap_params.need_dealloc = false;
        // }
      // #ifdef _DEBUGV2
        // LOG(INFO) << " swapout done. "<<(total_bytes>>20)<<" " << swap_params.tensor_name;
      // #endif
        swap_params.if_done = true;
      });
      
      DeallocateRaw(swap_params.swapped_gpu_buffer.first);
      LOG(INFO) << " swapout done. "<<(total_bytes>>20)<<" " << swap_params.tensor_name;

      swap_params.ram_or_gpu = true;
      // LOG(INFO)<<"SwapOut "<<tensor_name<<" with size M "<<(total_bytes>>20);//<<" "<<src_ptr<<" "<<cpu_part_dst_ptr;
      swap_params.tensor_buffer->SetMainName(tensor_name);
      return true;
}


void GPUBFCAllocator::SwapIn(const string& tensor_name) {
  // std::lock_guard<std::mutex> l(lock_);
  /* if (invalid_swap_.count(tensor_name) != 0) {
  #ifdef _DEBUGV2
    LOG(INFO) << "Ignore the invalid swap in: " << tensor_name;
  #endif
    return;
  } */

  DCHECK(tensor_swap_params_map_.count(tensor_name));
  auto& swap_params = tensor_swap_params_map_[tensor_name];
  // LOG(INFO)<<"try swapin "<<tensor_name;

  if(swap_params.tensor_buffer == nullptr){
    // LOG(INFO)<<"NULL "<<tensor_name;
    return;
  }
  // if(swap_params.tensor_buffer->data()==nullptr){//not in
  //   if(strcmp(tensor_name.c_str(),swap_params.tensor_buffer->GetMainName().c_str())!=0){
  //     LOG(INFO)<<"swap in shared "<<tensor_name<<" with main "<<swap_params.tensor_buffer->GetMainName();
  //     SwapIn(swap_params.tensor_buffer->GetMainName());
  //     LOG(INFO)<<"done";
  //     return;
  //   }
  // }else{//it is already in
  //   return;
  // }
  if(!swap_params.ram_or_gpu){//if in GPU, after get the lock
    return;
  }
  auto& cv_mu = swap_params.cv_mu;
  std::lock_guard<std::mutex> l(*(cv_mu.second));
  if (swap_params.data_ready.is_swapin() || swap_params.data_ready.is_in()) {
    return;
  }
  /* {
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    int ready = swap_params.data_ready;
    if (ready != SwapStatus::SWAPPING_OUT and ready != SwapStatus::OUT) {
      return;
    }
    swap_params.data_ready = SwapStatus::SWAPPING_IN;
  } */
  swap_params.data_ready.set_swapin();

// #ifdef _DEBUG
  // LOG(INFO) << "Start to SwapIn " << tensor_name;
// #endif

  void* gpu_part_src_ptr = swap_params.swapped_gpu_buffer.first;
  void* cpu_part_src_ptr = swap_params.swapped_cpu_buffer.first;
  int64 gpu_part_size = swap_params.swapped_gpu_buffer.second;
  int64 cpu_part_size = swap_params.swapped_cpu_buffer.second;

  Device* device = swap_params.device;
  DeviceContext* device_context = swap_params.device_context;
  const DeviceBase::GpuDeviceInfo* dev_info = nullptr;
  se::Stream* recv_stream = nullptr;

  // Get comp. stream and d2h stream to wait for to make sure right trigger time and ready data
  Status s = PrepareCopy(device, device_context, &dev_info, &recv_stream);
  auto device_to_host_stream =
    static_cast<const GPUDeviceContext*>(device_context)->device_to_host_stream();
  if (!s.ok()) {
    LOG(FATAL) << "PrepareCopy failed";
    // std::lock_guard<std::mutex> l(*(cv_mu.second));
    // swap_params.data_ready = SwapStatus::OUT;
    swap_params.data_ready.unset_swapin();
    return;
  }

  static Allocator* cuda_host_allocator = GPUProcessState::singleton()->GetCUDAHostAllocator(0);

  // TODO(px): deprecated: no need partial swapping
  if (gpu_part_size > 0) {
  // #ifdef _DEBUGV2
    LOG(INFO) << "[SwapIn] Start to try to merge copy.";
  // #endif
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(gpu_part_src_ptr);
    CHECK(h != kInvalidChunkHandle);
    BFCAllocator::Chunk* c = ChunkFromHandle(h);
    BFCAllocator::Chunk* c_next = nullptr;
    if (c->next != kInvalidChunkHandle) {
      c_next = ChunkFromHandle(c->next);
    }
    mutex_lock ll(BFCAllocator::lock_);
    if (c_next && c_next->size >= cpu_part_size && ! c_next->in_use()) {
      // try to avoid the intra-device memory copy

      RemoveFreeChunkFromBin(c->next);
      c_next->allocation_id = next_allocation_id_++;
      void* gpu_part2_ptr = c_next->ptr;

      auto host_to_device_stream =
        static_cast<const GPUDeviceContext*>(device_context)->host_to_device_stream();
      if (host_to_device_stream == nullptr) {
        LOG(FATAL) << "No host_to_device_stream is available.";
        // std::lock_guard<std::mutex> l(*(cv_mu.second));
        // swap_params.data_ready = SwapStatus::OUT;
        swap_params.data_ready.unset_swapin();
        return;
      }

      // TODO: we don't need to wait for the recv_stream as the data in host is ready for sure
      // (px): As this insertion is in stack, need to block the h2d stream
      // wait for comp. and d2h stream to make sure right trigger time and ready data
      // recv_stream .aka. compute stream, 
      //add a event after memcpy, and wait for it to finish, then execute
      host_to_device_stream->ThenWaitFor(recv_stream, device_to_host_stream);

      se::DeviceMemoryBase gpu_dst_ptr(gpu_part2_ptr, cpu_part_size);
      host_to_device_stream->ThenMemcpy(&gpu_dst_ptr, cpu_part_src_ptr, cpu_part_size);
      // dev_info->event_mgr->ThenRecordEvent(host_to_device_stream, &);

      dev_info->event_mgr->ThenExecute(
        host_to_device_stream,
        [this, host_to_device_stream, gpu_part_src_ptr, gpu_part2_ptr, cpu_part_src_ptr, &swap_params]() {
          if (!host_to_device_stream->ok()) {
            LOG(FATAL) << "CPU->GPU Memcpy failed";
            return;
          }
          auto& cv_mu = swap_params.cv_mu;
          std::lock_guard<std::mutex> l(*(cv_mu.second));
          // swap_params.data_ready = SwapStatus::IN;
          swap_params.data_ready.set_in();
          MergeBuffers(gpu_part_src_ptr, gpu_part2_ptr);
          swap_params.tensor_buffer->set_data(gpu_part_src_ptr);
          cuda_host_allocator->DeallocateRaw(cpu_part_src_ptr);
          cv_mu.first->notify_all();
        });

      return;
    }
    swap_params.ram_or_gpu = false;
  }

  void* dst_ptr = AllocateRaw(0, gpu_part_size + cpu_part_size);//warp this
  DCHECK(dst_ptr);

  // TODO(px): to be removed
  if (gpu_part_size > 0) {
  #ifdef _DEBUGV2
    LOG(INFO) << "[SwapIn] Start to device_to_device copy.";
  #endif
    auto device_to_device_stream =
      static_cast<const GPUDeviceContext*>(device_context)->device_to_device_stream(0);
    if (device_to_device_stream == nullptr) {
      LOG(FATAL) << "No device_to_device_stream is available.";
      // std::lock_guard<std::mutex> l(*(cv_mu.second));
      // swap_params.data_ready = SwapStatus::OUT;
      return;
    }

    se::DeviceMemoryBase gpu_src_ptr(gpu_part_src_ptr, gpu_part_size);
    se::DeviceMemoryBase gpu_dst_ptr(dst_ptr, gpu_part_size);

    device_to_device_stream->ThenMemcpy(&gpu_dst_ptr, gpu_src_ptr, gpu_part_size);

    dev_info->event_mgr->ThenExecute(
      device_to_device_stream,
      [this, gpu_part_src_ptr]() {
        DeallocateRaw(gpu_part_src_ptr);
      });
  }


  auto host_to_device_stream=
      static_cast<const GPUDeviceContext*>(device_context)->host_to_device_stream();

  if (host_to_device_stream == nullptr) {
    LOG(FATAL) << "No host_to_device_stream is available.";
    // std::lock_guard<std::mutex> l(*(cv_mu.second));
    // swap_params.data_ready = SwapStatus::OUT;
    return;
  }

  // Wait for the recv-stream to come to the appropriate trigger point
  host_to_device_stream->ThenWaitFor(recv_stream);

  se::DeviceMemoryBase gpu_dst_ptr((void*)((uintptr_t)dst_ptr + gpu_part_size), cpu_part_size);
  host_to_device_stream->ThenMemcpy(&gpu_dst_ptr, cpu_part_src_ptr, cpu_part_size);

  dev_info->event_mgr->ThenRecordEvent(host_to_device_stream, &swap_params.in_e);

  // set the status
  swap_params.need_in_addr = true;
  swap_params.in_gpu_src = dst_ptr;
  // swap_params.need_wait_in = true;
  swap_params.if_done_in = false;
  swap_params.tensor_buffer->set_data(dst_ptr); //no, we should set this, only when we swapped done
  // Use of the input may outlive stack scope, so keep a ref.
  dev_info->event_mgr->ThenExecute(
      host_to_device_stream,
      [host_to_device_stream, dst_ptr, cpu_part_src_ptr, &swap_params,cpu_part_size]() {
        if (!host_to_device_stream->ok()) {
          LOG(FATAL) << "GPU->CPU Memcpy failed";
          // std::lock_guard<std::mutex> l(*(swap_params.cv_mu.second));
          // swap_params.data_ready = SwapStatus::OUT;
          return;
        }
        auto &cv_mu = swap_params.cv_mu;
        std::lock_guard<std::mutex> l(*(cv_mu.second));
      // #ifdef _DEBUGV2
        LOG(INFO) << " swap in done. "<<(cpu_part_size>>20)<<" "<< swap_params.tensor_name;//<<" "<<(void*)&swap_params.if_done_in;
      // #endif
        // swap_params.data_ready = SwapStatus::IN;
        swap_params.data_ready.set_in();
        cuda_host_allocator->DeallocateRaw(cpu_part_src_ptr);
        //set the spin
        swap_params.if_done_in = true;
      });
}
}  // namespace tensorflow
