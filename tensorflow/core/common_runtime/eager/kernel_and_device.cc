/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/context.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

#include <fstream>
#include <string>

std::string GetEnv(const std::string& env_name) {
  const char* env = std::getenv(env_name.c_str());
  if (env == nullptr) return "";
  return env;
}
//TODO(RGY) change this to model dir!
const std::string innodes_file = "/tmp/tf_static_graph/1_innodes.txt";
const std::string outnodes_file = "/tmp/tf_static_graph/1_outnodes.txt";

static std::fstream tensor_access_fout;
static std::fstream fout_in(innodes_file.c_str(), fout_in.out);
static std::fstream fout_out(outnodes_file.c_str(), fout_out.out);


namespace tensorflow {

// static
Status KernelAndDevice::InitOp(Device* device, const NodeDef& ndef,
                               KernelAndDevice* out) {
  OpKernel* k = nullptr;
  Status s = CreateOpKernel(device->device_type().c_str(), device,
                            device->GetAllocator(AllocatorAttributes()),
                            nullptr, ndef, TF_GRAPH_DEF_VERSION, &k);
  out->device_ = device;
  out->kernel_.reset(k);
  out->flib_ = nullptr;
  out->runner_ = nullptr;
  out->default_runner_ = [](std::function<void()> f) { f(); };
  return s;
}

// static
Status KernelAndDevice::Init(const NodeDef& ndef, FunctionLibraryRuntime* flib,
                             std::function<void(std::function<void()>)>* runner,
                             KernelAndDevice* out) {
  OpKernel* k = nullptr;
  Status s = flib->CreateKernel(ndef, &k);
  out->device_ = flib->device();
  out->kernel_.reset(k);
  out->flib_ = flib;
  out->runner_ = runner;
  out->default_runner_ = [](std::function<void()> f) { f(); };
  //check if the file is valid
  if (!(fout_in.is_open() && fout_out.is_open()))
    LOG(FATAL) << "Can not open graph structure file";
  return s;
}

Status KernelAndDevice::Run(std::vector<Tensor>* inputs,
                            std::vector<Tensor>* outputs,
                            NodeExecStats* stats,
                            const std::string& op_uname) {
  ScopedStepContainer step_container(0, [this](const string& name) {
    device_->resource_manager()->Cleanup(name).IgnoreError();
  });
  return this->Run(&step_container, inputs, outputs, stats, op_uname);
}

Status KernelAndDevice::Run(ScopedStepContainer* step_container,
                            std::vector<Tensor>* inputs,
                            std::vector<Tensor>* outputs,
                            NodeExecStats* stats,
                            const std::string& op_uname) {
  gtl::InlinedVector<TensorValue, 4> input_vector;
  uint64 time_ = Env::Default()->NowMicros();
  static bool log_tensor_access = (GetEnv("TF_LOG_TENSOR_ACCESS") == "true") ? true : false;
  if (log_tensor_access) {
    if (!tensor_access_fout.is_open()) {
      tensor_access_fout.open("/tmp/tensor_access.txt", tensor_access_fout.out);
    }
  }
  if (stats != nullptr) {
    int i = 0;
    for (Tensor& t : *inputs) {
      if (!t.Name().empty()) ++i;
    }
    if (i) fout_in << "SrcNode" << "\t" << op_uname << "\t" << i << "\n";
  }
  for (Tensor& t : *inputs) {
    input_vector.push_back(TensorValue(&t));
    // CHECK(!t.Name().empty());
    std::string tensor_name = t.Name();    
    if (tensor_name.empty()) {
      // LOG(INFO) << op_uname << " with a empty tensor:";
      // LOG(INFO) << "Shape: " << t.shape().DebugString();
      // LOG(INFO) << "data: " << t.data();
      // LOG(INFO) << "buffer: " << t.buffer();
      // LOG(FATAL) << "Tensor name is empty!";
      continue;
    }
    // seems there is no op with empty name, but there are tensors with empty name
    bool is_anonymous = false;
    if (!tensor_name.substr(0, tensor_name.find_first_of('_')).compare(EagerContext::kanonymous_op_name)) {
      is_anonymous = true;
    }
    LOG(FATAL) << "RecordTensorAccess!";
    t.RecordTensorAccess(tensor_name, time_);
    if (stats != nullptr) {
      auto pos = tensor_name.find_first_of(':');
      std::string node_name = tensor_name.substr(0, pos);
      std::string slot_ = tensor_name.substr(pos+1, tensor_name.length());
      // int slot;
      // sscanf(slot_.c_str(), "%d", &slot);
      fout_in << "InputNode" << "\t" << node_name << "\t" << slot_ << "\n";
      if (is_anonymous) {
        fout_in << "\tShape: \t" << t.shape().DebugString() << "\n";
        fout_in << "\tdata: \t" << t.data() << "\n";
        fout_in << "\tbuffer: \t" << t.buffer() << "\n";
      }
      if (log_tensor_access) {
        if (!tensor_access_fout.is_open()) {
          LOG(FATAL) << "Failed to open /tmp/tensor_access.txt";
        }
        tensor_access_fout << tensor_name << "\t" << time_ << "\n";
      }
    }
  }

  std::vector<AllocatorAttributes> out_attrs(kernel_->num_outputs());
  for (size_t i = 0; i < out_attrs.size(); ++i) {
    out_attrs[i].set_on_host(kernel_->output_memory_types()[i] ==
                             tensorflow::HOST_MEMORY);
  }

  OpKernelContext::Params params;
  params.device = device_;
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &input_vector;
  params.op_kernel = kernel_.get();
  params.resource_manager = device_->resource_manager();
  params.output_attr_array = gtl::vector_as_array(&out_attrs);
  params.function_library = flib_;
  params.slice_reader_cache = &slice_reader_cache_;
  params.rendezvous = rendez_;
  params.cancellation_manager = &cm_;
  if (stats != nullptr) {
    params.track_allocations = true;
  }
  if (runner_ == nullptr) {
    params.runner = &default_runner_;
  } else {
    params.runner = runner_;
  }

  params.step_container = step_container;

  OpKernelContext context(&params);

  if (kernel_->def().op() == "_Recv") {
    // TODO(apassos) do not special-case _Recv. Currently the GPU device fails
    // if trying to run _Recv->Compute(), specifically checking for _Recv. To go
    // around this we call _Recv->ComputeAsync, to mimic graph mode behavior.
    AsyncOpKernel* async = kernel_->AsAsync();
    Notification done;
    device_->ComputeAsync(async, &context, [&done]() { done.Notify(); });
    done.WaitForNotification();
  } else {
    device_->Compute(kernel_.get(), &context);
  }
  if (!context.status().ok()) return context.status();

  outputs->clear();
  DeviceContext* dev_ctx = context.op_device_context();
  CHECK(dev_ctx);
  /* if (dev_ctx == nullptr) {
    // TODO(px): if this happen, get from device
    LOG(FATAL) << "Can not get DeviceContext from OpKernelContext.";
  } */
  for (int i = 0; i < context.num_outputs(); ++i) {
    outputs->push_back(Tensor(*context.mutable_output(i)));
    if (op_uname.empty()) continue;
    std::string tensor_name = op_uname + ":" + std::to_string(i);
    outputs->back().RecordSwapContext({tensor_name, device_, dev_ctx});
  }
  if (stats != nullptr) {
    for (const auto& allocator_pair : context.wrapped_allocators()) {
      AllocatorMemoryUsed* memory = stats->add_memory();
      memory->set_allocator_name(allocator_pair.first->Name());
      auto sizes = allocator_pair.second->GetSizes();
      memory->set_total_bytes(std::get<0>(sizes));
      memory->set_peak_bytes(std::get<1>(sizes));
      memory->set_live_bytes(std::get<2>(sizes));

      AllocatorStats allocator_stats;
      allocator_pair.first->GetStats(&allocator_stats);
      memory->set_allocator_bytes_in_use(allocator_stats.bytes_in_use);
      allocator_pair.second->GetRecordsAndUnRef();
    }
    auto* ms = stats->mutable_memory_stats();
    ms->set_temp_memory_size(context.temp_memory_allocated());
    for (const auto& alloc_id : context.persistent_alloc_ids()) {
      ms->mutable_persistent_tensor_alloc_ids()->Add(alloc_id);
    }

    ms->set_persistent_memory_size(context.persistent_memory_allocated());
  }
  return Status::OK();
}

}  // namespace tensorflow
