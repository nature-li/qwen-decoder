#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BUILD_DIR=$SCRIPT_DIR/build
MODEL=$SCRIPT_DIR/qwen2.5-3b-instruct-fp16.gguf

PROMPT0="how are you"
PROMPT1="who are you"
PROMPT2="tell me a joke"
PROMPT3="hi"

echo "=== GPU v7 (batch_size=4) ==="
printf "%s\n%s\n%s\n%s\n" "$PROMPT0" "$PROMPT1" "$PROMPT2" "$PROMPT3" \
    | $BUILD_DIR/gpu_decoder_v7 $MODEL 4