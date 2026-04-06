from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class Phi3Config:
    # Default points to workspace root: /home/.../comp560-LamPham/Phi-3-mini-4k-instruct
    model_path: str = "../Phi-3-mini-4k-instruct"
    device_map: Optional[str] = "auto"
    trust_remote_code: bool = True
    use_4bit: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


class Phi3Local:
    def __init__(self, config: Optional[Phi3Config] = None):
        self.config = config or Phi3Config()
        self.repo_root = Path(__file__).resolve().parents[1]
        self.model_dir = self._resolve_model_dir(self.config.model_path)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}. "
                "Put/download the model into ../Phi-3-mini-4k-instruct or pass an absolute model_path"
            )

        quantization_config = None
        torch_dtype = None

        if self.config.use_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("use_4bit=True requires CUDA. Disable 4-bit on CPU.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        load_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "quantization_config": quantization_config,
            "torch_dtype": torch_dtype,
        }

        # Some accelerate/transformers combos incorrectly call .to() on 4-bit models
        # when device_map is provided, so keep it unset for quantized single-GPU runs.
        if self.config.device_map and not self.config.use_4bit:
            load_kwargs["device_map"] = self.config.device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            **load_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=self.config.trust_remote_code,
        )

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        gen_len = max_new_tokens or self.config.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_len,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.generate(prompt)
        return full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text

    def _resolve_model_dir(self, model_path: str) -> Path:
        path = Path(model_path)
        if path.is_absolute():
            return path
        return (self.repo_root / path).resolve()
