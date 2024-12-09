import gc
import numpy as np
import os
import time
from contextlib import nullcontext

import tiktoken
import torch
import torch.nn.functional as F

from model import GPT
from quantized_model import Quantized_GPT

# -----------------------------------------------------------------------------
init_from = 'gpt2' # gpt2 variant (e.g. 'gpt2-xl')
max_new_tokens = 50 # number of tokens generated in each sample
temperature = 0.4 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability

seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

dataset = 'wikitext'
block_size = 1024
num_warmup = 1
speculative_tokens = 3
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Measure perplexity
def measure_perplexity(model, data, batch_size):
    nll_weights = []
    nlls = []

    for i in range(0, len(data), block_size * batch_size):
        j = min(i + block_size * batch_size, len(data))
        ix = torch.arange(i, j, block_size)
        x = []
        for k in ix:
            x.append(torch.from_numpy((data[k : k + block_size]).astype(np.int64)))
        
        x = torch.stack([F.pad(y, (0, block_size - len(y)), value=-1) for y in x])
        nll_weights.append((x != -1).sum().item() / len(data))
        if device_type == 'cuda':
            # pin array x which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        with torch.no_grad():
            with ctx:
                y = x.clone()
                x[x == -1] = 0
                logits, _ = model(x, y)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = y[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
                nlls.append(loss)
    
    nlls = [nll_weights[i] * nlls[i] for i in range(len(nlls))]
    return torch.exp(torch.stack(nlls).sum()).item()

print("\nInference without quantization:\n")
# Load pre-trained model
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
model.eval()
torch.cuda.reset_peak_memory_stats(device=device)
model.to(device)
print(f"GPU memory allocated => {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
if compile:
    model = torch.compile(model)

# Run evaluation
torch.cuda.reset_peak_memory_stats(device=device)
perplexity_4 = measure_perplexity(model, val_data, batch_size=4)
print(f"GPT-2 perplexity on {dataset}/val.bin for a batch size of 4 => {perplexity_4:.4f}")
print(f"GPU memory allocated => {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB\n")

perplexity_12 = measure_perplexity(model, val_data, batch_size=12)
print(f"GPT-2 perplexity on {dataset}/val.bin for a batch size of 12 => {perplexity_12:.4f}")
print(f"GPU memory allocated => {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB\n")

# Free up space
del model
gc.collect()
torch.cuda.empty_cache()

print("\nInference with quantization:\n")
# Load pre-trained model and quantize
q_model = Quantized_GPT.from_pretrained(init_from, dict(dropout=0.0))
q_model.quantize()
torch.cuda.reset_peak_memory_stats(device=device)
q_model.to(device)
print(f"GPU memory allocated => {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
if compile:
    q_model = torch.compile(q_model)

# Run evaluation
torch.cuda.reset_peak_memory_stats(device=device)
perplexity_4 = measure_perplexity(q_model, val_data, batch_size=4)
print(f"Quantized GPT-2 perplexity on {dataset}/val.bin for a batch size of 4 => {perplexity_4:.4f}")
print(f"GPU memory allocated => {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB\n")

perplexity_12 = measure_perplexity(q_model, val_data, batch_size=12)
print(f"Quantized GPT-2 perplexity on {dataset}/val.bin for a batch size of 12 => {perplexity_12:.4f}")
print(f"GPU memory allocated => {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB\n")


print("\nInference latency for Standard decoding versus Speculative decoding:\n")
target = 'gpt2-medium'
draft = 'gpt2'

# Load target model
print(f"Loading target model M => {target}\n")
target_model = GPT.from_pretrained(target, dict(dropout=0.0))
target_model.eval()
target_model.to(device)
if compile:
    target_model = torch.compile(target_model)

# Load draft model
print(f"Loading draft model D => {draft}")
draft_model = GPT.from_pretrained(draft, dict(dropout=0.0))
draft_model.eval()
draft_model.to(device)
if compile:
    draft_model = torch.compile(draft_model)

start_idx = torch.randint(len(val_data) - block_size, (1,)).item()
start_ids = list(val_data[start_idx : start_idx + block_size])

def measure_inference_latency(target_model, draft_model=None, batch_size=1):
    x = torch.tensor(start_ids * batch_size, dtype=torch.long, device=device).reshape(batch_size, -1)
    with torch.no_grad():
        with ctx:
            t0 = time.time()
            torch.manual_seed(1337)
            if draft_model is not None:
                y = target_model.generate_speculative(x, max_new_tokens, draft_model, temperature=temperature, top_k=top_k, num_speculative=speculative_tokens)
            else:
                y = target_model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            t1 = time.time()
    dt = t1 - t0
    print(f"For batch size => {batch_size}, Inference Latency => {dt / max_new_tokens:.4f} sec/tok ({max_new_tokens} tokens, {block_size} context size)")
    return y[0].tolist()

print(f"\nStandard decoding with M ({target})")
target_generations = measure_inference_latency(target_model)
print(f"\nSpeculative decoding with M ({target}) and D ({draft})")
speculative_generations = measure_inference_latency(target_model, draft_model)
num_matching = len([1 for i in range(len(target_generations)) if target_generations[i] == speculative_generations[i]])

# Free up space
del target_model
del draft_model
gc.collect()
torch.cuda.empty_cache()

print("\nInference latency for Standard decoding versus Speculative decoding with quantiztion:\n")
target = 'gpt2'
draft = 'gpt2'

# Load target model
print(f"Loading target model M => {target}\n")
target_model = GPT.from_pretrained(target, dict(dropout=0.0))
target_model.eval()
target_model.to(device)
if compile:
    target_model = torch.compile(target_model)

# Load draft model and quantize
print(f"Loading draft model D => {draft}")
draft_model = Quantized_GPT.from_pretrained(draft, dict(dropout=0.0))
print(f"Quantizing draft model D ({draft})")
draft_model.quantize()
draft_model.eval()
draft_model.to(device)
if compile:
    draft_model = torch.compile(draft_model)

print(f"\nStandard decoding with M ({target})")
target_generations = measure_inference_latency(target_model)
print(f"\nSpeculative decoding with M ({target}) and quantized D ({draft})")
speculative_generations = measure_inference_latency(target_model, draft_model)

# Free up space
del target_model
del draft_model
gc.collect()
torch.cuda.empty_cache()


print("\nMemory usage for inference without quantization:\n")
# Load pre-trained model
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
model.eval()
torch.cuda.reset_peak_memory_stats(device=device)
model.to(device)
print(f"\nGPU memory allocated => {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
if compile:
    model = torch.compile(model)

# Run evaluation
torch.cuda.reset_peak_memory_stats(device=device)
perplexity_1 = measure_perplexity(model, val_data, batch_size=1)
print(f"GPT-2 perplexity on {dataset}/val.bin for a batch size of 1 => {perplexity_1:.4f}")
print(f"GPU memory allocated => {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB\n")
del model
gc.collect()
torch.cuda.empty_cache()

print("\nMemory usage for inference with quantization:\n")
# Load pre-trained model and quantize
q_model = Quantized_GPT.from_pretrained(init_from, dict(dropout=0.0))
q_model.quantize()
torch.cuda.reset_peak_memory_stats(device=device)
q_model.to(device)
print(f"\nGPU memory allocated => {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
if compile:
    q_model = torch.compile(q_model)

# Run evaluation
torch.cuda.reset_peak_memory_stats(device=device)
perplexity_1 = measure_perplexity(q_model, val_data, batch_size=1)
print(f"Quantized GPT-2 perplexity on {dataset}/val.bin for a batch size of 1 => {perplexity_1:.4f}")
print(f"GPU memory allocated => {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB\n")
del q_model
gc.collect()
torch.cuda.empty_cache()