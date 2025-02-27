import json
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Literal

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Config, Block
from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir, quantization
from scripts.prepare_alpaca import generate_prompt


lora_r = 8
lora_alpha = 16
lora_dropout = 0.05


def main(
    # prompt: str = "What food do lamas eat?",
    # input: str = "",
    lora_path: Path = Path("out/lora/alpaca/lit_model_lora_finetuned.pth"),
    checkpoint_dir: Path = Path(f"checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Literal["llm.int8", "gptq.int4"] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    strategy: str = "auto",
    devices: int = 1,
    precision: str = "bf16-true",
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    if strategy == "fsdp":
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, cpu_offload=False)
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, **json.load(fp))

    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.time()
    with fabric.init_module(empty_init=True), quantization(quantize):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.time()
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(lora_path) as lora_checkpoint:
        checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
        model.load_state_dict(checkpoint, strict=quantize is None)
    fabric.print(f"Time to load the model weights: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json", checkpoint_dir / "tokenizer_config.json")


    while(True):
        query = input("Enter question:\n")
        if not query.strip():
            break        
        sample = {"instruction": query, "input": ""}
        prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, device=model.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        t0 = time.perf_counter()
        y = generate(
            model,
            encoded,
            max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id,
        )
        t = time.perf_counter() - t0

        model.reset_cache()
        output = tokenizer.decode(y)
        output = output.split("### Response:")[1].strip()
        fabric.print(output)

        tokens_generated = y.size(0) - prompt_length
        fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
        if fabric.device.type == "cuda":
            fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    CLI(main)
