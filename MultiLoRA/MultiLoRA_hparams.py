from dataclasses import dataclass
from typing import List, Literal

from util.hparams import HyperParams


@dataclass
class MultiLoRAHyperParams(HyperParams):
    # Method
    model_name: str
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    nullspace_threshold: float
    L2: float
    
    # MultiLoRA specific
    lora_rank: int = 64  # LoRA rank r
    
    # Routing parameters (PDF §1.3)
    routing_mode: Literal["none", "hard", "soft", "cw"] = "none"  # none=sum all, hard=argmax, soft=softmax, cw=confidence-weighted
    router_rank: int = 16  # U_j subspace dimension for routing
    router_gamma: float = 10.0  # softmax temperature
    router_preload_gpu: bool = False  # Preload routing tensors to GPU (fast, high VRAM)
    router_chunk_edits: int = 8  # Chunk CW edits on GPU (0 = disabled)
    
    # CW-Edit parameters (idea_3)
    router_tau_floor: float = 1e-6  # Minimum value for tau
    router_neg_quantile: float = 0.95  # Quantile for negative calibration threshold
    router_top_m: int = 0  # Top-m gating (0 = disabled)
    router_use_neg_calib: bool = True  # Use negative calibration
    router_score_norm: bool = True  # L2 normalize k before computing scores
    router_kbank_max: int = 1024  # Maximum columns in K_bank cache