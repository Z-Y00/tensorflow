#!/bin/bash
cd /home/rgy/CLionProjects/tf-benchmark/benchmarks/scripts/tf_cnn_benchmarks/
# python tf_cnn_benchmarks.py --help
# exit
#/home/rgy/CLionProjects/tf-benchmark/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py
# export TF_LOG_TENSOR_ACCESS=true
# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_MIN_VLOG_LEVEL=3
# export TF_SWAPPING_OPTION=0
# const std::string swap_policy_env = "SWAP_POLICY_FILE";
# static bool log_tensor_access = (GetEnv("TF_LOG_TENSOR_ACCESS") == "true") ? true : false;
# valgrind python tf_cnn_benchmarks.py --num_gpus=1 --variable_update=parameter_server \
  #  --model=alexnet  > log 2>&1
# exit
# tensorboard --logdir ./tf
#  pip install tensorboard==1.12.0
# --summary_verbosity=3\
#  --train_dir=/tmp/tf/\
#  --trace_file=/tmp/tf/trace \
#  --save_summaries_steps=1 \
config="tf_cnn_benchmarks.py \
 -gpu_memory_frac_for_testing=0.6 \
--num_batches=1\
 --variable_update=parameter_server --memory_optimization=NO_MEM_OPT --num_gpus=1"

#vgg16 baseline 39 LRU 
# gdb -ex r --args \
python  $config  \
  --batch_size=45 --model=vgg16 \
  > log 2>&1
exit

# resnet50 baseline 43
# python  $config  \
    #  --batch_size=43 --model=resnet50 
exit

# resnet152 baseline 18
python $config   \
      --batch_size=18 --model=resnet152
exit


python  $config  --batch_size=32 --model=inception3

python  $config --batch_size=16 --model=inception4 