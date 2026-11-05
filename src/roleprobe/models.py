"""Model registry and per-role chat formats."""
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer

ROLES = ('system', 'user', 'tool', 'cot', 'assistant')


@dataclass(frozen=True)
class ChatFormat:
    """Per-role (prefix, suffix) text wrappers; prefixes may reference {tool_name}.

    Wrapper/content boundaries must fall on special tokens for token-exact role
    labels (true for Harmony; ChatML is close enough for smoke tests).
    """
    wrap: dict


HARMONY = ChatFormat(wrap={
    'system':    ('<|start|>system<|message|>', '<|end|>'),
    'user':      ('<|start|>user<|message|>', '<|end|>'),
    'tool':      ('<|start|>functions.{tool_name} to=assistant<|channel|>commentary<|message|>', '<|end|>'),
    'cot':       ('<|start|>assistant<|channel|>analysis<|message|>', '<|end|>'),
    'assistant': ('<|start|>assistant<|channel|>final<|message|>', '<|end|>'),
    'tool_call': ('<|start|>assistant<|channel|>commentary to=functions.{tool_name}<|message|>', '<|call|>'),
})

CHATML_THINK = ChatFormat(wrap={
    'system':    ('<|im_start|>system\n', '<|im_end|>\n'),
    'user':      ('<|im_start|>user\n', '<|im_end|>\n'),
    'tool':      ('<|im_start|>user\n<tool_response>\n', '\n</tool_response><|im_end|>\n'),
    'cot':       ('<|im_start|>assistant\n<think>\n', '\n</think>\n\n<|im_end|>\n'),
    'assistant': ('<|im_start|>assistant\n', '<|im_end|>\n'),
    'tool_call': ('<|im_start|>assistant\n<tool_call>\n', '\n</tool_call><|im_end|>\n'),
})


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    fmt: ChatFormat
    n_layers: int
    attn: str | None = None
    C: float = 5e-3       # inverse L2 strength, sklearn convention
    scale: bool = False   # standardize features before the probe


MODELS = {
    'gptoss-20b':  ModelSpec('openai/gpt-oss-20b', HARMONY, 24,
                             attn='kernels-community/vllm-flash-attn3'),
    'gptoss-120b': ModelSpec('openai/gpt-oss-120b', HARMONY, 36,
                             attn='kernels-community/vllm-flash-attn3', scale=True),
    'qwen3-0.6b':  ModelSpec('Qwen/Qwen3-0.6B', CHATML_THINK, 28, C=1e-1),  # smoke tests only
}


def default_layers(n_layers):
    return list(range(0, n_layers, 4 if n_layers >= 30 else 2))


def load_model(spec, device):
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype='auto', attn_implementation=spec.attn).to(device).eval()
    return tok, model
