#!/usr/bin/env sh

EXAMPLE=$DEEP_OCR_ROOT
DATA=/workspace/caffe_dataset
TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/caffe train --solver=$DEEP_OCR_ROOT/data/caffe_nets/lower_eng/lenet_solver.prototxt $@
