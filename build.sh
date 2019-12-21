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
tf_ver="1.11.0"
# dst_dir="latest_pkg"
dst_dir="tensorflow_pkg"

bazel build ${dbg_opt} --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 

./bazel-bin/tensorflow/tools/pip_package/build_pip_package ${HOME}/tf/tmp/${dst_dir}
echo y > ${HOME}/tf/y
pip uninstall tensorflow < ${HOME}/tf/y
pip install ${HOME}/tf/tmp/${dst_dir}/tensorflow-${tf_ver}-cp27-cp27mu-linux_x86_64.whl
