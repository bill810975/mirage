from dataclasses import dataclass, field
from typing import Literal, Optional

from ..kernel import get_key_paths, KNGraph, TBGraph
from ..core import *


@dataclass
class SpecDecodeConfig:
    method: str

@dataclass
class LookaheadConfig(SpecDecodeConfig):
    method: Literal["lookahead"] = "lookahead"
    spec_length: int = 7

@dataclass
class PromptLookupConfig(SpecDecodeConfig):
    method: Literal["promptlookup"] = "promptlookup"
    ngram_size: int = 3
    spec_length: int = 5

@dataclass
class MTPConfig(SpecDecodeConfig):
    """Multi-Token Prediction config for DeepSeek V3 (vLLM-compatible naming).

    Supports 0-7 draft tokens + 1 bonus token per decode step.
    """
    method: Literal["mtp"] = "mtp"
    # Core MTP parameters (vLLM: num_speculative_tokens)
    num_speculative_tokens: int = 1
    # Number of MTP predictor layers (DeepSeek V3: num_nextn_predict_layers)
    num_mtp_layers: int = 1
    # Verification mode (vLLM: rejection_sample_method)
    rejection_sample_method: str = "strict"  # "strict" | "probabilistic" | "synthetic"
    # Synthetic mode param (vLLM: synthetic_acceptance_rate)
    synthetic_acceptance_rate: Optional[float] = None

    def __post_init__(self):
        assert 1 <= self.num_speculative_tokens <= 7, \
            f"num_speculative_tokens must be 1-7, got {self.num_speculative_tokens}"
        assert self.rejection_sample_method in ("strict", "probabilistic", "synthetic"), \
            f"Invalid rejection_sample_method: {self.rejection_sample_method}"
        if self.rejection_sample_method == "synthetic":
            assert self.synthetic_acceptance_rate is not None, \
                "synthetic_acceptance_rate required for synthetic mode"

def spec_decode_class(spec_decode: str,
                      ngram_size: int = 3,
                      spec_length: int = 5,
                      num_speculative_tokens: int = 1,
                      rejection_sample_method: str = "strict",
                      synthetic_acceptance_rate: float = None):
    if spec_decode == "lookahead":
        return LookaheadConfig(spec_length=spec_length)
    elif spec_decode == "promptlookup":
        return PromptLookupConfig(ngram_size=ngram_size, spec_length=spec_length)
    elif spec_decode == "mtp":
        return MTPConfig(
            num_speculative_tokens=num_speculative_tokens,
            rejection_sample_method=rejection_sample_method,
            synthetic_acceptance_rate=synthetic_acceptance_rate,
        )
    elif spec_decode is None:
        return None
    else:
        raise NotImplementedError(f"Spec decode method {spec_decode} not implemented")