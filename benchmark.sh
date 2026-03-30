#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BUILD_DIR=$SCRIPT_DIR/build
MODEL=$SCRIPT_DIR/qwen2.5-3b-instruct-fp16.gguf
PROMPT="请把下面这段文字翻译成英文：人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"

echo "=== CPU ==="
echo "$PROMPT" | $BUILD_DIR/cpu_decoder $MODEL

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