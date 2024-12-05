import os
import time
import pickle
from contextlib import nullcontext

import tiktoken
import torch

from model import GPTConfig, GPT


def inference(model):
    init_from = 'gpt2'
    out_dir = 'out'
    start = "\n"
    num_samples = 1
    max_new_tokens = 500
    temperature = 0.8
    top_k = 200 
    seed = 1337
    device = 'cuda' 
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
    compile = False 
    exec(open('configurator.py').read())

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)

        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'

        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)

    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi = meta['stoi']
        encode = lambda s: [stoi[c] for c in s]
    else:
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)

    def measure_inference_latency(batch_size):
        x = torch.tensor(start_ids * batch_size, dtype=torch.long, device=device).reshape(batch_size, -1)
        with torch.no_grad():
            with ctx:
                tokens_decoded = 0
                start_time = time.time()
                for _ in range(num_samples):
                    output = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    tokens_decoded += output.shape[0] * (output.shape[1] - 1)
                end_time = time.time()
        inference_time = end_time - start_time
        tokens_per_second = tokens_decoded / inference_time
        print(f"For batch size => {batch_size}, Inference Latency => {tokens_per_second:.4f} tokens/second")

    measure_inference_latency(batch_size=1)
    measure_inference_latency(batch_size=12)