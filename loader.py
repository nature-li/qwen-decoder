from gguf import GGUFReader
import numpy as np

reader = GGUFReader("qwen2.5-0.5b-instruct-fp16.gguf")
for tensor in reader.tensors:
    if tensor.name == "token_embd.weight":
        data = tensor.data.flatten()
        print(f"token_embedding[0][0] = {float(data[0])}")
        print(f"token_embedding[0][1] = {float(data[1])}")
    if tensor.name == "output_norm.weight":
        data = tensor.data.flatten()
        print(f"rms_final[0] = {float(data[0])}")
        print(f"rms_final[1] = {float(data[1])}")
for key, val in reader.fields.items():
    if key == "tokenizer.ggml.tokens":
        tokens = [str(bytes(val.parts[i]), encoding='utf-8', errors='replace') 
                  for i in val.data]
        print(f"vocab[0]      = {tokens[0]}")
        print(f"vocab[100]    = {tokens[100]}")
        print(f"vocab[1000]   = {tokens[1000]}")
        print(f"vocab[151643] = {tokens[151643]}")
        print(f"vocab[151645] = {tokens[151645]}")
        break