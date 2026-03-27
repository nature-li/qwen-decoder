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