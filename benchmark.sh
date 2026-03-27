#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BUILD_DIR=$SCRIPT_DIR/build
MODEL=$SCRIPT_DIR/qwen2.5-1.5b-instruct-fp16.gguf
PROMPT="who are you"

echo "=== CPU ==="
echo "$PROMPT" | $BUILD_DIR/cpu_decoder $MODEL

echo ""
echo "=== GPU v1 ==="
echo "$PROMPT" | $BUILD_DIR/gpu_decoder_v1 $MODEL