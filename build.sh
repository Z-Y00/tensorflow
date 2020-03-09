#!/bin/bash

dbg_opt=""
if [ $# == 1 ];then
  if [ $1 == "-g" ];then
    dbg_opt="--copt=-g -c dbg"
  else
    echo "Unrecognized params: $1"
    exit
  fi
fi
dbg_opt="--copt=-g  " #-c dbg"

tf_ver="1.11.0"
# dst_dir="latest_pkg"
dst_dir="tensorflow_pkg"
# dst_dir="baseline_pkg"
# bazel test -c opt --config=cuda //tensorflow/...
# --local_ram_resources=10240    --host_jvm_args=-Xms512m --ram_utilization_factor 50 
bazel build  ${dbg_opt}  --verbose_failures --jobs=6 --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package  || exit 1

./bazel-bin/tensorflow/tools/pip_package/build_pip_package ${HOME}/tf/tmp/${dst_dir}
echo y > ${HOME}/tf/y
pip uninstall tensorflow < ${HOME}/tf/y
pip install ${HOME}/tf/tmp/${dst_dir}/tensorflow-${tf_ver}-cp27-cp27mu-linux_x86_64.whl
# pip install ${HOME}/tf/tmp/baseline_pkg/tensorflow-${tf_ver}-cp27-cp27mu-linux_x86_64.whl

./benchmark.sh