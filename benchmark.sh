#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BUILD_DIR=$SCRIPT_DIR/build
MODEL=$SCRIPT_DIR/qwen2.5-3b-instruct-fp16.gguf
PROMPT="who are you? could you tell me a joke?"

# echo "=== CPU ==="
# echo "$PROMPT" | $BUILD_DIR/cpu_decoder $MODEL

echo ""
echo "=== GPU v1 ==="
echo "$PROMPT" | $BUILD_DIR/gpu_decoder_v1 $MODEL

echo ""
echo "=== GPU v2 ==="
echo "$PROMPT" | $BUILD_DIR/gpu_decoder_v2 $MODEL

echo ""
echo "=== GPU v3 ==="
echo "$PROMPT" | $BUILD_DIR/gpu_decoder_v3 $MODEL


echo ""
echo "=== GPU v4 ==="
echo "$PROMPT" | $BUILD_DIR/gpu_decoder_v4 $MODEL

echo ""
echo "=== GPU v5 ==="
echo "$PROMPT" | $BUILD_DIR/gpu_decoder_v5 $MODEL