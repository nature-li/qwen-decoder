# 生成一个很长的 prompt
python -c "print('请把下面这段文字翻译成英文：' + '人工智能是计算机科学的重要分支。' * 50)" | ./gpu_decoder_v5 ../qwen2.5-3b-instruct-fp16.gguf

# 生成一个很长的 prompt
python -c "print('请把下面这段文字翻译成英文：' + '人工智能是计算机科学的重要分支。' * 50)" | ./gpu_decoder_v6 ../qwen2.5-3b-instruct-fp16.gguf