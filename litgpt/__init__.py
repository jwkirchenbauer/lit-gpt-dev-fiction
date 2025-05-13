# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import logging
import re

# from litgpt.model import GPT # Avoid generically importing model.py
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt import optim
from litgpt import settings
from litgpt import monitor

# Suppress excessive warnings, see https://github.com/pytorch/pytorch/issues/111632
pattern = re.compile(".*Profiler function .* will be ignored")
logging.getLogger("torch._dynamo.variables.torch").addFilter(lambda record: not pattern.search(record.getMessage()))

# Avoid printing state-dict profiling output at the WARNING level when saving a checkpoint
logging.getLogger("torch.distributed.fsdp._optim_utils").disabled = True
logging.getLogger("torch.distributed.fsdp._debug_utils").disabled = True

# Avoid generically importing model.py
# __all__ = ["GPT", "Config", "Tokenizer", "optim", "settings", "monitor"]
__all__ = ["Config", "Tokenizer", "optim", "settings", "monitor"]
