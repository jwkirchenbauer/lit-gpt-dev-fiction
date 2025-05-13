# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from pathlib import Path
from typing import Optional, Union
import os

import torch


class Tokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            try:  # Try downloading the missing tokeinizer
                from transformers import AutoTokenizer, LlamaTokenizerFast

                temp_processor = AutoTokenizer.from_pretrained(checkpoint_dir)
                assert isinstance(temp_processor, LlamaTokenizerFast)
                assert isinstance(temp_processor.vocab_file, str)
                checkpoint_dir = Path(os.path.dirname(temp_processor.vocab_file))
            except (OSError, AssertionError):
                raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        self.bos_id = None
        self.eos_id = None
        self.pad_id = None

        self.cache_token_id = None
        self.eod_token_id = None

        if (checkpoint_dir / "tokenizer.json").is_file():
            from transformers import AutoTokenizer

            self.processor = AutoTokenizer.from_pretrained(
                str(checkpoint_dir), add_bos_token=False, add_eos_token=False
            )

            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                self.bos_id = self.processor.bos_token_id
                self.eos_id = self.processor.eos_token_id
                self.pad_id = self.processor.pad_token_id
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
                if self.pad_id is None:
                    self.pad_id = config.get("pad_token_id")  # idk if this will always work
        elif "open_llama" in str(checkpoint_dir):
            from transformers import LlamaTokenizer

            self.processor = LlamaTokenizer.from_pretrained(
                str(checkpoint_dir), add_bos_token=False, add_eos_token=False
            )

            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                self.bos_id = self.processor.bos_token_id
                self.eos_id = self.processor.eos_token_id
                self.pad_id = self.processor.pad_token_id
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
                if self.pad_id is None:
                    self.pad_id = config.get("pad_token_id")  # idk if this will always work
        else:
            raise NotImplementedError("Couldn't load a tokenizer from the given checkpoint directory.")

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size

    def __len__(self) -> int:
        # https://stackoverflow.com/questions/67412925/what-is-the-difference-between-lentokenizer-and-tokenizer-vocab-size#:~:text=Size%20of%20the%20base%20vocabulary%20(without%20the%20added%20tokens).&text=So%20you%20can%20clearly%20see,plus%20the%20len(added_tokens_encoder)%20.
        return len(self.processor)

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        tokens = self.processor.encode(string)

        if bos:
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined a bos token")
            tokens = [bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor, skip_special_tokens: bool = False) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens, skip_special_tokens=skip_special_tokens)
