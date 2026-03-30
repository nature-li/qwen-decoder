#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BUILD_DIR=$SCRIPT_DIR/build
MODEL=$SCRIPT_DIR/qwen2.5-3b-instruct-fp16.gguf

echo "=== GPU v8 (max_batch=4, 8 requests) ==="
printf "who are you\ntell me a joke\nhi\nwhat is 1+1\n河北有什么好吃的\nhow to learn cuda\nwhat is AI\n写一首诗\n\n" \
    | $BUILD_DIR/gpu_decoder_v8 $MODEL 4