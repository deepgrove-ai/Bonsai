<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->
<p align="center">
  <img src="figs/bonsai.png" width="200" alt="Bonsai Logo">

<h3 align="center" style="font-size: 30px">Bonsai: A Small Ternary-Weight Language Model</h3>
</p>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/deepgrove/Bonsai"><b>🤗 Model</b></a> 
  <a href="https://github.com/deepgrove-ai/Bonsai/tree/main/paper/Bonsai.pdf"><b>📄 Paper</b></a> 
</div>



<!-- # Bonsai -->

## Introduction

Bonsai is a small 500 million parameter ternary weight language model trained by deepgrove. Bonsai adopts the Llama architecture and Mistral tokenizer following [Danube 3](https://arxiv.org/pdf/2407.09276v1), with modified linear layers to support ternary weights. The model has been trained primarily using DCLM-Pro and Fineweb-Edu. Bonsai marks a new paradigm of efficiency, being trained in less than 5 billion tokens.

## Results

Bonsai achieves competitive performance among its peers, being one of the first ternary models to do so. Evalution results are below; for more detailed results and comparisons to other ternary models, please see the accompanying paper linked above. We use lm-eval for all benchmarks outside of MMLU and lighteval's cloze formulation for MMLU.

<div align="center">

| Model | ARC-c | ARC-e | HS. | OBQA | PiQA | Wino. | MMLU | Avg |
|-------|--------|--------|------|-------|-------|--------|-------|-----|
| MobiLlama 0.5B | 26.62 | 46.68 | 51.66 | 30.00 | 71.65 | 54.50 | 28.61 | 44.25 |
| Qwen 2 0.5B | 28.84 | 50.29 | 49.12 | 33.00 | 69.26 | 56.99 | 31.78 | 45.61 |
| MobileLLM 600M | 29.01 | 56.65 | 55.35 | 34.00 | 71.65 | 59.75 | 31.40 | 48.13 |
| Qwen 2.5 0.5B | 32.25 | 58.29 | 52.18 | 35.40 | 69.91 | 56.12 | 33.40 | 48.22 |
| **Bonsai** | 33.36 | 57.95 | 48.04 | 34.00 | 70.24 | 54.85 | 30.28 | 46.96 |

</div>

## Usage
Bonsai can be easily used through the Huggingface Transformers library. However, we note that all operations are currently performed in 16 bit precision; we're currently working towards integrating our model design with custom mixed precision kernels. A quick example follows:

```{python}
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("hespere-ai/Bonsai", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("hespere-ai/Bonsai", trust_remote_code=True)
text = "What is the capital of France?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
We note that Bonsai is not instruction tuned; we highly recommend finetuning the model before usage in a downstream task.


