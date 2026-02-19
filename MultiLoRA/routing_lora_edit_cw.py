import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .MultiLoRA_hparams import MultiLoRAHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


# ============================================================
# Routing Functions (PDF §1.3) - OPTIMIZED
# ============================================================

def compute_router_subspace(K_j: torch.Tensor, router_rank: int) -> torch.Tensor:
    """
    Compute subspace basis U_j for routing.
    
    PDF Eq.18-19:
    - K_j = U @ Σ @ V.T
    - U_j = U[:, :r_route]  (d0 × r_route)
    
    Args:
        K_j: Key vectors for edit j, shape (d0 × u)
        router_rank: Number of singular vectors to keep
    
    Returns:
        U_j: Subspace basis, shape (d0 × r_route)
    """
    # Handle case where K_j has fewer columns than router_rank
    r = min(router_rank, K_j.shape[1])
    
    # SVD of K_j
    U, S, _ = torch.linalg.svd(K_j, full_matrices=False)
    
    # Keep top-r singular vectors
    U_j = U[:, :r]
    
    return U_j


def compute_routing_scores_vectorized(
    k: torch.Tensor,
    U_stacked: torch.Tensor,
    normalize_k: bool = False,
) -> torch.Tensor:
    """
    Compute routing scores with vectorized operations (FAST).
    
    PDF Eq.21:
    s_i(k) = ||U_i.T @ k||²
    
    Args:
        k: Input key vector, shape (batch, d0)
        U_stacked: Stacked subspace bases, shape (n_edits, d0, r_route)
        normalize_k: Whether to L2 normalize k before computing scores
    
    Returns:
        scores: Routing scores, shape (batch, n_edits)
    """
    # Optionally normalize k
    if normalize_k:
        k = F.normalize(k, p=2, dim=-1)
    
    # k: (batch, d0) -> (batch, 1, d0)
    # U_stacked: (n_edits, d0, r_route) -> (1, n_edits, d0, r_route)
    # proj = k @ U_j for each edit: (batch, n_edits, r_route)
    proj = torch.einsum('bd,ndr->bnr', k, U_stacked)
    # scores = ||proj||^2 along r_route dimension
    scores = (proj ** 2).sum(dim=-1)  # (batch, n_edits)
    return scores


def compute_routing_scores(
    k: torch.Tensor,
    U_bank: List[torch.Tensor],
    normalize_k: bool = False,
) -> torch.Tensor:
    """
    Compute routing scores for input key k (legacy interface).
    
    PDF Eq.21:
    s_i(k) = ||U_i.T @ k||² = k.T @ P_i @ k
    
    Args:
        k: Input key vector, shape (d0,) or (batch, d0)
        U_bank: List of subspace bases U_j, each shape (d0 × r_j)
        normalize_k: Whether to L2 normalize k before computing scores
    
    Returns:
        scores: Routing scores, shape (n_edits,) or (batch, n_edits)
    """
    if len(U_bank) == 0:
        return torch.tensor([])
    
    # Handle batched input
    is_batched = k.dim() == 2
    if not is_batched:
        k = k.unsqueeze(0)  # (1, d0)
    
    # Optionally normalize k
    if normalize_k:
        k = F.normalize(k, p=2, dim=-1)
    
    scores = []
    for U_j in U_bank:
        U_j = U_j.to(k.device)
        # ||U_j.T @ k||² for each k in batch
        proj = k @ U_j  # (batch, r_j)
        score = (proj ** 2).sum(dim=-1)  # (batch,)
        scores.append(score)
    
    scores = torch.stack(scores, dim=-1)  # (batch, n_edits)
    
    if not is_batched:
        scores = scores.squeeze(0)  # (n_edits,)
    
    return scores


def compute_cw_gates(
    scores: torch.Tensor,
    t: torch.Tensor,
    tau: torch.Tensor,
    top_m: int = 0,
) -> torch.Tensor:
    """
    Compute confidence-weighted gates using sigmoid.
    
    Args:
        scores: Routing scores, shape (batch, n_edits) or (n_edits,)
        t: Threshold values, shape (n_edits,)
        tau: Temperature values, shape (n_edits,)
        top_m: Top-m gating (0 = disabled)
    
    Returns:
        gates: Gate values, shape (batch, n_edits) or (n_edits,)
    """
    # Handle batched vs non-batched
    is_batched = scores.dim() == 2
    if not is_batched:
        scores = scores.unsqueeze(0)  # (1, n_edits)
    
    # Expand t and tau to match scores shape
    t = t.unsqueeze(0).to(scores.device)  # (1, n_edits)
    tau = tau.unsqueeze(0).to(scores.device)  # (1, n_edits)
    
    # Compute gates: sigmoid((scores - t) / tau)
    gates = torch.sigmoid((scores - t) / tau)  # (batch, n_edits)
    
    # Apply top-m if enabled
    if top_m > 0 and top_m < gates.shape[-1]:
        # Get top-m indices for each sample
        _, top_indices = torch.topk(gates, k=top_m, dim=-1)  # (batch, top_m)
        # Create mask
        mask = torch.zeros_like(gates)
        mask.scatter_(-1, top_indices, 1.0)
        gates = gates * mask
    
    if not is_batched:
        gates = gates.squeeze(0)  # (n_edits,)
    
    return gates


def route_and_aggregate(
    k: torch.Tensor,
    B_bank: List[torch.Tensor],
    U_bank: List[torch.Tensor],
    A: torch.Tensor,
    routing_mode: str = "soft",
    gamma: float = 10.0,
    # CW-Edit parameters
    stats_bank: Optional[List[Dict]] = None,
    hparams: Optional[MultiLoRAHyperParams] = None,
) -> torch.Tensor:
    """
    Route input k and compute aggregated delta.
    
    PDF Eq.17:
    f(k) = (W + Σ φ_j(k) · B_j @ A) @ k
    
    Args:
        k: Input key vector, shape (d0,) or (batch, d0)
        B_bank: List of B matrices, each shape (d1 × r)
        U_bank: List of subspace bases U_j, each shape (d0 × r_route)
        A: Nullspace basis, shape (r × d0)
        routing_mode: "none", "hard", "soft", or "cw"
        gamma: Temperature for soft routing
        stats_bank: List of stats dicts for CW-Edit, each with {"mu", "tau", "t"}
        hparams: Hyperparameters (needed for CW-Edit)
    
    Returns:
        delta: Aggregated update, shape (d1 × d0)
    """
    n_edits = len(B_bank)
    
    if n_edits == 0:
        return torch.zeros(A.shape[1], A.shape[1], device=A.device)
    
    A = A.to(k.device)
    
    if routing_mode == "none":
        # Sum all B @ A
        delta = sum(B.to(k.device) @ A for B in B_bank)
    
    elif routing_mode == "hard":
        # Hard routing: select one edit
        scores = compute_routing_scores(k, U_bank)
        if scores.dim() == 1:
            selected = scores.argmax().item()
            delta = B_bank[selected].to(k.device) @ A
        else:
            # Batched: use most common selection
            selected = scores.mean(dim=0).argmax().item()
            delta = B_bank[selected].to(k.device) @ A
    
    elif routing_mode == "soft":
        # Soft routing: weighted sum
        scores = compute_routing_scores(k, U_bank)
        
        # Handle batched scores by averaging
        if scores.dim() == 2:
            scores = scores.mean(dim=0)
        
        # Softmax with temperature
        weights = F.softmax(gamma * scores, dim=-1)
        
        # Weighted sum of B @ A
        delta = sum(w * B.to(k.device) @ A for w, B in zip(weights, B_bank))
    
    elif routing_mode == "cw":
        # CW-Edit: confidence-weighted gating
        if hparams is None:
            raise ValueError("hparams required for CW-Edit routing_mode")
        if stats_bank is None or len(stats_bank) != n_edits:
            raise ValueError(f"stats_bank required for CW-Edit, got {len(stats_bank) if stats_bank else 0} stats for {n_edits} edits")
        
        # Compute scores with optional normalization
        normalize_k = hparams.router_score_norm
        scores = compute_routing_scores(k, U_bank, normalize_k=normalize_k)
        
        # Extract t and tau from stats_bank
        t = torch.tensor([stat["t"] for stat in stats_bank], device=k.device, dtype=k.dtype)
        tau = torch.tensor([stat["tau"] for stat in stats_bank], device=k.device, dtype=k.dtype)
        
        # Compute gates
        gates = compute_cw_gates(scores, t, tau, top_m=hparams.router_top_m)
        
        # Handle batched gates by averaging
        if gates.dim() == 2:
            gates = gates.mean(dim=0)
        
        # Weighted sum: Σ gate_j * B_j @ A
        delta = sum(g * B.to(k.device) @ A for g, B in zip(gates, B_bank))
    
    else:
        raise ValueError(f"Unknown routing_mode: {routing_mode}")
    
    return delta


# ============================================================
# Stats Computation for CW-Edit
# ============================================================

def compute_positive_scores(K_j: torch.Tensor, U_j: torch.Tensor, normalize_k: bool = True) -> torch.Tensor:
    """
    Compute positive scores s+ from current edit's K_j.
    
    Args:
        K_j: Key vectors for edit j, shape (d0, u)
        U_j: Subspace basis, shape (d0, r_route)
        normalize_k: Whether to L2 normalize k before computing scores
    
    Returns:
        s_plus: Positive scores, shape (u,)
    """
    # K_j: (d0, u), U_j: (d0, r_route)
    # For each column k in K_j: s = ||U_j.T @ k||^2
    if normalize_k:
        K_j = F.normalize(K_j, p=2, dim=0)  # Normalize each column
    
    proj = U_j.T @ K_j  # (r_route, u)
    s_plus = (proj ** 2).sum(dim=0)  # (u,)
    return s_plus


def compute_negative_scores(K_bank: torch.Tensor, U_j: torch.Tensor, normalize_k: bool = True) -> torch.Tensor:
    """
    Compute negative scores s- from other edits' K samples.
    
    Args:
        K_bank: Cached K samples from other edits, shape (d0, n_samples)
        U_j: Subspace basis, shape (d0, r_route)
        normalize_k: Whether to L2 normalize k before computing scores
    
    Returns:
        s_minus: Negative scores, shape (n_samples,)
    """
    if K_bank.shape[1] == 0:
        return torch.tensor([])
    
    if normalize_k:
        K_bank = F.normalize(K_bank, p=2, dim=0)  # Normalize each column
    
    proj = U_j.T @ K_bank  # (r_route, n_samples)
    s_minus = (proj ** 2).sum(dim=0)  # (n_samples,)
    return s_minus


def compute_stats_for_edit(
    K_j: torch.Tensor,
    U_j: torch.Tensor,
    K_bank: Optional[torch.Tensor],
    hparams: MultiLoRAHyperParams,
) -> Dict[str, float]:
    """
    Compute stats (mu, tau, t) for a single edit.
    
    Args:
        K_j: Key vectors for current edit, shape (d0, u)
        U_j: Subspace basis, shape (d0, r_route)
        K_bank: Cached K samples from other edits, shape (d0, n_samples) or None
        hparams: Hyperparameters
    
    Returns:
        stats: Dict with "mu", "tau", "t"
    """
    normalize_k = hparams.router_score_norm
    
    # Compute s+ from current edit
    s_plus = compute_positive_scores(K_j, U_j, normalize_k=normalize_k)
    mu = s_plus.mean().item()
    tau = s_plus.std().item()
    
    # Apply tau floor
    tau = max(tau, hparams.router_tau_floor)
    
    # Compute t_j
    if hparams.router_use_neg_calib and K_bank is not None and K_bank.shape[1] > 0:
        # Compute s- from other edits
        s_minus = compute_negative_scores(K_bank, U_j, normalize_k=normalize_k)
        if len(s_minus) > 0:
            # t_j = quantile(s-)
            t = torch.quantile(s_minus, hparams.router_neg_quantile).item()
        else:
            # Fallback: t_j = mu
            t = mu
    else:
        # Fallback: t_j = mu
        t = mu
    
    return {"mu": mu, "tau": tau, "t": t}


# ============================================================
# K_bank Cache Management
# ============================================================

class KBankCache:
    """
    FIFO cache for K samples from other edits (stored on CPU).
    """
    
    def __init__(self, max_cols: int = 1024):
        self.max_cols = max_cols
        self.K_bank = None  # (d0, n_cols) on CPU
    
    def add_samples(self, K_new: torch.Tensor):
        """
        Add new K samples to the cache (FIFO).
        
        Args:
            K_new: New K samples, shape (d0, u) on any device
        """
        K_new_cpu = K_new.cpu()  # Move to CPU
        
        if self.K_bank is None:
            # Initialize with first samples
            self.K_bank = K_new_cpu
        else:
            # Concatenate
            self.K_bank = torch.cat([self.K_bank, K_new_cpu], dim=1)
            
            # Trim if exceeds max_cols (FIFO: keep last max_cols)
            if self.K_bank.shape[1] > self.max_cols:
                self.K_bank = self.K_bank[:, -self.max_cols:]
    
    def get_bank(self) -> Optional[torch.Tensor]:
        """Get the cached K bank."""
        return self.K_bank


# ============================================================
# Main Functions
# ============================================================

def get_nullspace_A(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    hparams: MultiLoRAHyperParams,
) -> torch.Tensor:
    """
    Compute nullspace basis A for LoRA editing.
    
    PDF Eq.6-7:
    - SVD(K0 @ K0.T) = U @ Λ @ U.T
    - A ≈ U_{d0-r+1:d0}.T  (shape: r × d0)
    
    Returns:
        A: torch.Tensor of shape (r, d0), where r = nullspace_dim
    """
    # Get covariance matrix (K0 @ K0.T approximation)
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples,
        hparams.mom2_dtype,
    ).cpu()
    
    # SVD to find nullspace
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    
    # Find indices with small singular values (nullspace directions)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    
    if len(small_singular_indices) == 0:
        raise ValueError(f"No nullspace found with threshold {threshold}. "
                        f"Consider increasing threshold.")
    
    # A = U[:, nullspace_indices].T  (shape: nullspace_dim × d0)
    A = U[:, small_singular_indices].T
    
    print(f"Layer {layer}: nullspace dim = {len(small_singular_indices)}, A shape = {A.shape}")
    return A


def apply_Routing_LoRA_Edit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MultiLoRAHyperParams,
    cache_template: Optional[str] = None,
    A_matrices: Dict[int, torch.Tensor] = None,
    B_bank: Dict[int, List[torch.Tensor]] = None,
    U_bank: Dict[int, List[torch.Tensor]] = None,  # NEW: for routing
    router_stats_bank: Dict[int, List[Dict]] = None,  # NEW: for CW-Edit stats
    K_bank_cache: Dict[int, KBankCache] = None,  # NEW: for K_bank cache
) -> Tuple[AutoModelForCausalLM, Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]], Dict[int, List[Dict]]]:
    """
    Executes the MultiLoRA editing algorithm with optional routing.
    
    PDF Eq.9-11:
    - X = A @ K_j           # (r × u)
    - M = X @ X.T + A @ A.T # (r × r)  
    - B_j = E @ X.T @ M⁻¹   # (d1 × r)
    - Δ = B_j @ A           # (d1 × d0)
    
    Args:
        A_matrices: Pre-computed nullspace basis A for each layer
        B_bank: Accumulated B matrices from previous edits
        U_bank: Accumulated subspace bases for routing (NEW)
        router_stats_bank: Accumulated stats for CW-Edit (NEW)
        K_bank_cache: K_bank cache for negative calibration (NEW)
    
    Returns:
        model: Updated model
        B_bank: Updated B_bank with new B_j matrices
        U_bank: Updated U_bank with new U_j matrices
        router_stats_bank: Updated router_stats_bank with new stats
    """
    # Initialize banks if not provided
    if U_bank is None:
        U_bank = {layer: [] for layer in hparams.layers}
    if router_stats_bank is None:
        router_stats_bank = {layer: [] for layer in hparams.layers}
    if K_bank_cache is None:
        K_bank_cache = {layer: KBankCache(max_cols=hparams.router_kbank_max) for layer in hparams.layers}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MultiLoRA request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        if not data_loaded:
            # MEMORY-EFFICIENT: Clear GPU cache before compute_z
            torch.cuda.empty_cache()
            
            # NOTE: compute_z needs gradients, so don't use torch.no_grad()
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )
            # Store on CPU to save GPU memory (detach to remove from computation graph)
            z_list.append(cur_z.detach().cpu())

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, v_star=cur_z.detach().cpu().numpy())
                print(f"Cached k/v pair at {cache_fname}")
            
            # MEMORY-EFFICIENT: Clear GPU cache after compute_z
            torch.cuda.empty_cache()
    
    # Move zs to GPU when needed
    zs = torch.stack([z.cuda() if z.device.type == 'cpu' else z for z in z_list], dim=1)
    
    # MEMORY-EFFICIENT: Clear GPU cache before starting layer processing
    torch.cuda.empty_cache()

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        
        # MEMORY-EFFICIENT: Clear GPU cache before forward pass
        torch.cuda.empty_cache()

        # Get current model activations: K_j shape (d0 × u)
        # Use torch.no_grad() to avoid storing gradients (no optimization needed here)
        model.eval()  # Set to eval mode for inference
        with torch.no_grad():
            layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")
        
        # MEMORY-EFFICIENT: Clear GPU cache after compute_ks
        torch.cuda.empty_cache()

        # Compute residual error
        with torch.no_grad():
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
        
        # MEMORY-EFFICIENT: Clear GPU cache after get_module_input_output_at_words
        torch.cuda.empty_cache()
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())
        
        # MEMORY-EFFICIENT: Clear cur_zs after use
        del cur_zs
        torch.cuda.empty_cache()

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)
        
        # MEMORY-EFFICIENT: Clear targets after use
        del targets
        torch.cuda.empty_cache()
        
        # ===== PDF Eq.9-11: Compute B_j =====
        # MEMORY-EFFICIENT: Keep A on CPU, compute A @ A.T on CPU
        A_cpu = A_matrices[layer]  # (r, d0) on CPU
        K = layer_ks  # (d0 × u) on GPU
        
        # MEMORY-EFFICIENT: Compute A @ A.T on CPU first (large matrix)
        # A @ A.T is (r, r) which can be very large (1-3GB for GPT-J)
        print(f"Computing A @ A.T on CPU (shape: {A_cpu.shape[0]} x {A_cpu.shape[0]})...")
        AAT_cpu = A_cpu @ A_cpu.T  # (r, r) on CPU
        
        # Move A to GPU temporarily for X = A @ K computation
        A = A_cpu.cuda()  # (r, d0) on GPU
        X = A @ K  # (r × u) on GPU
        A = A.cpu()  # Move back to CPU immediately
        del A_cpu
        torch.cuda.empty_cache()
        
        # Move AAT to GPU for M computation
        AAT = AAT_cpu.cuda()  # (r, r) on GPU
        del AAT_cpu
        
        # M = X @ X.T + A @ A.T  (r × r)
        XXT = X @ X.T  # (r, r)
        M = XXT + AAT + hparams.L2 * torch.eye(AAT.shape[0], device="cuda")
        
        # Clean up intermediate tensors
        del XXT, AAT
        torch.cuda.empty_cache()
        
        # B_j = E @ X.T @ M^{-1}  (d1 × r)
        B_j = torch.linalg.solve(M.T, (resid @ X.T).T).T
        
        # Store B_j in B_bank
        B_bank[layer].append(B_j.cpu())
        
        # ===== NEW: Compute and store U_j for routing =====
        U_j = compute_router_subspace(K.cpu(), hparams.router_rank)
        U_bank[layer].append(U_j)
        print(f"U_j shape: {U_j.shape} (router_rank={hparams.router_rank})")
        
        # ===== NEW: Compute and store stats for CW-Edit =====
        if hparams.routing_mode == "cw":
            # Get K_bank from cache (other edits' samples)
            K_bank = K_bank_cache[layer].get_bank()
            
            # Compute stats
            stats = compute_stats_for_edit(K.cpu(), U_j, K_bank, hparams)
            router_stats_bank[layer].append(stats)
            
            # Log stats
            print(f"[CW-Edit] Stats for edit {len(router_stats_bank[layer])-1}: "
                  f"mu={stats['mu']:.6f}, tau={stats['tau']:.6f}, t={stats['t']:.6f}")
            
            # Add current K to cache (for future edits)
            K_bank_cache[layer].add_samples(K.cpu())
        
        # ===== Apply update based on routing_mode =====
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        
        if hparams.routing_mode == "none":
            # Original behavior: directly add B_j @ A to weights
            # MEMORY-EFFICIENT: Move A to GPU temporarily for B_j @ A
            A_gpu = A_matrices[layer].cuda()  # (r, d0) on GPU
            upd_matrix = B_j @ A_gpu  # (d1, d0)
            A_gpu = A_gpu.cpu()  # Move back to CPU
            del A_gpu
            torch.cuda.empty_cache()
            
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
                
            print("orig norm", torch.linalg.norm(weights[weight_name]))
            print("upd norm", torch.linalg.norm(upd_matrix))
            print(f"B_j shape: {B_j.shape}, A shape: {A_matrices[layer].shape}")
            
            with torch.no_grad():
                weights[weight_name][...] = weights[weight_name] + upd_matrix
            
            # Clean up
            del upd_matrix
            torch.cuda.empty_cache()
        else:
            # Routing mode: don't modify weights, just store B_j and U_j
            # The routing will be applied at inference time
            print(f"[routing_mode={hparams.routing_mode}] B_j stored, weights NOT modified")
            print(f"B_j shape: {B_j.shape}, A shape: {A_matrices[layer].shape}")
        
        # Clear GPU memory (cur_zs and targets already deleted above)
        del layer_ks, X, M, B_j
        # B_j is already moved to CPU in B_bank, but delete GPU copy
        torch.cuda.empty_cache()

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, B_bank, U_bank, router_stats_bank


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


# ============================================================
# Routing Hooks for Inference (PDF §1.3 Eq.17) - OPTIMIZED
# ============================================================

class RoutingLoRAHooks:
    """
    Forward hooks for applying routing-based LoRA edits at inference time.
    
    PDF Eq.17:
    f(k) = (W + Σ φ_j(k) · B_j @ A) @ k
    
    When routing_mode != "none", the weights are NOT modified during editing.
    Instead, this hook applies the routed delta dynamically at inference.
    
    MEMORY OPTIMIZATION:
    - Delta_j = B_j @ A is NOT pre-computed (too memory-intensive for GPT-J)
    - Instead, we compute it on-the-fly: (B_j @ A) @ k = B_j @ (A @ k)
    - This uses much less memory: O(r * d0) instead of O(d1 * d0) per edit
    - U_bank is stacked for vectorized score computation
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        hparams: MultiLoRAHyperParams,
        A_matrices: Dict[int, torch.Tensor],
        B_bank: Dict[int, List[torch.Tensor]],
        U_bank: Dict[int, List[torch.Tensor]],
        router_stats_bank: Optional[Dict[int, List[Dict]]] = None,  # NEW: for CW-Edit
        device: str = "cuda",
    ):
        self.model = model
        self.hparams = hparams
        self.hooks = []
        self.device = device
        
        # ===== Routing tensor placement =====
        self.preload_gpu = getattr(hparams, "router_preload_gpu", True)
        chunk_size = getattr(hparams, "router_chunk_edits", 0)
        if hparams.routing_mode == "cw":
            if chunk_size <= 0:
                # Only force chunking if preload is not explicitly enabled (for evaluation speed)
                if not self.preload_gpu:
                    chunk_size = 8
                    print("[RoutingLoRAHooks] router_chunk_edits not set; forcing chunked CW routing (8).")
                else:
                    chunk_size = 0  # Disable chunking if preload is enabled
            # Only force preload_gpu=False if chunking is enabled
            if chunk_size > 0:
                self.preload_gpu = False
        self.use_chunking = (hparams.routing_mode == "cw" and not self.preload_gpu and chunk_size > 0)
        
        if self.preload_gpu:
            print("[RoutingLoRAHooks] Setting up performance-optimized routing (pre-loading to GPU)...")
        elif self.use_chunking:
            print(f"[RoutingLoRAHooks] Setting up chunked routing (chunk_size={chunk_size}, no stacking)...")
        else:
            print("[RoutingLoRAHooks] Setting up memory-efficient routing (CPU-stacked, GPU on-demand)...")
        
        self.A_matrices = {}     # {layer: A (r, d0) on GPU}
        self.B_stacked = {}      # {layer: (n_edits, d1, r) stacked (GPU if preloaded, else CPU)} - None if chunking
        self.U_stacked = {}      # {layer: (n_edits, d0, r_route) stacked (GPU if preloaded, else CPU)} - None if chunking
        self.n_edits = {}        # {layer: n_edits}
        
        # For chunking mode: store B_bank/U_bank directly (no stacking)
        if self.use_chunking:
            self.B_bank = B_bank  # {layer: List[B_j]}
            self.U_bank = U_bank  # {layer: List[U_j]}
        
        # NEW: For CW-Edit
        self.t_stacked = {}      # {layer: (n_edits,) on GPU or CPU}
        self.tau_stacked = {}    # {layer: (n_edits,) on GPU or CPU}
        
        for layer in hparams.layers:
            n = len(B_bank[layer])
            self.n_edits[layer] = n
            
            if n == 0:
                continue
            
            # Pre-load A to GPU (used in every forward pass)
            try:
                self.A_matrices[layer] = A_matrices[layer].to(device)  # (r, d0) on GPU
            except torch.cuda.OutOfMemoryError as e:
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                print(f"ERROR: Out of memory while loading A matrix for layer {layer}")
                print(f"  GPU memory: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB allocated, "
                      f"{torch.cuda.memory_reserved(device)/1024**3:.2f}GB reserved")
                raise
            
            # Stack B_j for vectorized operations (skip if chunking mode)
            if not self.use_chunking:
                try:
                    if self.preload_gpu:
                        B_list = [B.to(device) for B in B_bank[layer]]
                    else:
                        B_list = [B.cpu() for B in B_bank[layer]]
                    self.B_stacked[layer] = torch.stack(B_list, dim=0)  # (n_edits, d1, r)
                    del B_list  # Free intermediate list memory
                except torch.cuda.OutOfMemoryError:
                    if device.startswith('cuda'):
                        torch.cuda.empty_cache()
                    print(f"ERROR: Out of memory while loading B matrices for layer {layer} ({n} edits)")
                    print(f"  GPU memory: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB allocated, "
                          f"{torch.cuda.memory_reserved(device)/1024**3:.2f}GB reserved")
                    raise
            else:
                self.B_stacked[layer] = None  # Chunking mode: no stacking
            
            # Stack U matrices (skip if chunking mode)
            if not self.use_chunking:
                try:
                    if self.preload_gpu:
                        U_list = [U.to(device) for U in U_bank[layer]]
                    else:
                        U_list = [U.cpu() for U in U_bank[layer]]
                    self.U_stacked[layer] = torch.stack(U_list, dim=0)  # (n_edits, d0, r_route)
                    del U_list  # Free intermediate list memory
                except torch.cuda.OutOfMemoryError:
                    if device.startswith('cuda'):
                        torch.cuda.empty_cache()
                    print(f"ERROR: Out of memory while loading U matrices for layer {layer} ({n} edits)")
                    print(f"  GPU memory: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB allocated, "
                          f"{torch.cuda.memory_reserved(device)/1024**3:.2f}GB reserved")
                    raise
            else:
                self.U_stacked[layer] = None  # Chunking mode: no stacking
            
            # NEW: Pre-load stats for CW-Edit
            if hparams.routing_mode == "cw" and router_stats_bank is not None:
                stats_list = router_stats_bank.get(layer, [])
                if len(stats_list) == n:
                    t_list = [stat["t"] for stat in stats_list]
                    tau_list = [stat["tau"] for stat in stats_list]
                    stats_device = device if self.preload_gpu else "cpu"
                    self.t_stacked[layer] = torch.tensor(t_list, device=stats_device, dtype=torch.float32)
                    self.tau_stacked[layer] = torch.tensor(tau_list, device=stats_device, dtype=torch.float32)
                else:
                    # Fallback: use mu for t, floor for tau
                    print(f"Warning: stats mismatch for layer {layer}, using fallback")
                    stats_device = device if self.preload_gpu else "cpu"
                    self.t_stacked[layer] = torch.zeros(n, device=stats_device, dtype=torch.float32)
                    self.tau_stacked[layer] = torch.full((n,), hparams.router_tau_floor, device=stats_device, dtype=torch.float32)
            
            # Memory usage info
            r = self.A_matrices[layer].shape[0]
            d0 = self.A_matrices[layer].shape[1]
            if self.use_chunking:
                d1 = self.B_bank[layer][0].shape[0]
                r_route = self.U_bank[layer][0].shape[1]
                chunk_size = getattr(self.hparams, "router_chunk_edits", 8)
                mem_per_chunk = (chunk_size * d1 * r + chunk_size * d0 * r_route) * 4 / (1024**3)  # GB per chunk
                print(f"  Layer {layer}: {n} edits, chunked mode (chunk_size={chunk_size}), "
                      f"~{mem_per_chunk:.2f}GB per chunk on GPU (no stacking)")
            else:
                d1 = self.B_stacked[layer].shape[1]
                mem_used = (n * d1 * r + r * d0 + n * d0 * self.U_stacked[layer].shape[2]) * 4 / (1024**3)  # GB
                print(f"  Layer {layer}: {n} edits, GPU memory: ~{mem_used:.2f}GB "
                      f"(A: {r}×{d0}, B: {n}×{d1}×{r}, U: {n}×{d0}×{self.U_stacked[layer].shape[2]})")
        
        print(f"[RoutingLoRAHooks] Setup done. Total edits per layer: {self.n_edits}")
        
    def _make_hook(self, layer: int):
        """Create a forward hook for a specific layer (MEMORY-EFFICIENT)."""
        
        def hook_fn(module, input, output):
            """
            Hook function that adds routed delta to the output.
            
            MEMORY-EFFICIENT: Computes Delta on-the-fly using (B_j @ A) @ k = B_j @ (A @ k)
            This avoids storing large Delta matrices (d1 × d0) on GPU.
            """
            if self.n_edits.get(layer, 0) == 0:
                return output
            
            # Get input to the layer (k vectors)
            inp = input[0]  # (batch, seq, d0)
            batch_size, seq_len, d0 = inp.shape
            inp_flat = inp.view(-1, d0)  # (batch * seq, d0)
            n_tokens = batch_size * seq_len
            
            # Get pre-loaded tensors (GPU if preloaded, else CPU and moved on-demand)
            A = self.A_matrices[layer]  # (r, d0) on GPU
            r = A.shape[0]
            
            # For chunking mode, we'll get B/U from B_bank/U_bank directly
            if self.use_chunking:
                # Get dimensions from first edit
                n_edits = self.n_edits[layer]
                d1 = self.B_bank[layer][0].shape[0]  # (d1, r)
            else:
                B_stacked = self.B_stacked[layer]
                U = self.U_stacked[layer]
                if not self.preload_gpu and B_stacked.device.type != "cuda":
                    B_stacked = B_stacked.to(self.device, non_blocking=True)
                    U = U.to(self.device, non_blocking=True)
                n_edits = B_stacked.shape[0]
                d1 = B_stacked.shape[1]
            
            # PERFORMANCE-OPTIMIZED: Compute A @ k first (vectorized)
            # A @ k: (r, d0) @ (n_tokens, d0).T -> (r, n_tokens)
            Ak = A @ inp_flat.T  # (r, n_tokens)
            
            if self.hparams.routing_mode == "none":
                # Sum all B_j @ (A @ k) = (sum B_j) @ (A @ k)
                B_sum = B_stacked.sum(dim=0)  # (d1, r)
                delta_output = (B_sum @ Ak).T  # (n_tokens, d1)
            
            elif self.hparams.routing_mode == "hard":
                # Compute routing scores (vectorized)
                scores = compute_routing_scores_vectorized(inp_flat, U)  # (n_tokens, n_edits)
                selected = scores.argmax(dim=-1)  # (n_tokens,)
                
                # PERFORMANCE-OPTIMIZED: Compute all deltas at once, then select
                # B_stacked: (n_edits, d1, r), Ak: (r, n_tokens)
                # einsum: 'edr,rt->etd' -> (n_edits, n_tokens, d1)
                all_deltas = torch.einsum('edr,rt->etd', B_stacked, Ak)  # (n_edits, n_tokens, d1)
                
                # Select the correct edit for each token
                # selected: (n_tokens,) -> gather along dim 0
                selected_expanded = selected.unsqueeze(0).unsqueeze(-1).expand(1, -1, d1)  # (1, n_tokens, d1)
                delta_output = all_deltas.gather(0, selected_expanded).squeeze(0)  # (n_tokens, d1)
            
            elif self.hparams.routing_mode == "soft":
                # Compute routing scores and weights (vectorized)
                scores = compute_routing_scores_vectorized(inp_flat, U)  # (n_tokens, n_edits)
                weights = F.softmax(self.hparams.router_gamma * scores, dim=-1)  # (n_tokens, n_edits)
                
                # PERFORMANCE-OPTIMIZED: Compute all deltas at once, then weighted sum
                # B_stacked: (n_edits, d1, r), Ak: (r, n_tokens)
                # einsum: 'edr,rt->etd' -> (n_edits, n_tokens, d1)
                all_deltas = torch.einsum('edr,rt->etd', B_stacked, Ak)  # (n_edits, n_tokens, d1)
                
                # Weighted sum: weights (n_tokens, n_edits), all_deltas (n_edits, n_tokens, d1)
                # einsum: 'te,etd->td' -> (n_tokens, d1)
                delta_output = torch.einsum('te,etd->td', weights, all_deltas)
            
            elif self.hparams.routing_mode == "cw":
                # CW-Edit: confidence-weighted gating
                # Compute routing scores with optional normalization
                normalize_k = self.hparams.router_score_norm
                chunk_size = getattr(self.hparams, "router_chunk_edits", 8)
                if self.use_chunking:
                    # Chunked CW routing: load from B_bank/U_bank directly (no stacking)
                    delta_output = torch.zeros(n_tokens, d1, device=self.device, dtype=Ak.dtype)
                    t_all = self.t_stacked[layer]
                    tau_all = self.tau_stacked[layer]
                    for start in range(0, n_edits, chunk_size):
                        end = min(start + chunk_size, n_edits)
                        # Load chunk from B_bank/U_bank directly
                        B_chunk_list = [self.B_bank[layer][i].to(self.device, non_blocking=True) for i in range(start, end)]
                        U_chunk_list = [self.U_bank[layer][i].to(self.device, non_blocking=True) for i in range(start, end)]
                        B_chunk = torch.stack(B_chunk_list, dim=0)  # (chunk, d1, r)
                        U_chunk = torch.stack(U_chunk_list, dim=0)  # (chunk, d0, r_route)
                        del B_chunk_list, U_chunk_list
                        
                        t_chunk = t_all[start:end].to(self.device, non_blocking=True)
                        tau_chunk = tau_all[start:end].to(self.device, non_blocking=True)
                        scores = compute_routing_scores_vectorized(
                            inp_flat, U_chunk, normalize_k=normalize_k
                        )  # (n_tokens, chunk)
                        gates = torch.sigmoid(
                            (scores - t_chunk.unsqueeze(0)) / tau_chunk.unsqueeze(0)
                        )  # (n_tokens, chunk)
                        if self.hparams.router_top_m > 0 and self.hparams.router_top_m < n_edits:
                            _, top_indices = torch.topk(
                                gates, k=min(self.hparams.router_top_m, gates.shape[1]), dim=-1
                            )
                            mask = torch.zeros_like(gates)
                            mask.scatter_(-1, top_indices, 1.0)
                            gates = gates * mask
                        all_deltas = torch.einsum('edr,rt->etd', B_chunk, Ak)
                        delta_output = delta_output + torch.einsum('te,etd->td', gates, all_deltas)
                        del B_chunk, U_chunk, t_chunk, tau_chunk, scores, gates, all_deltas
                        if self.device.startswith("cuda"):
                            torch.cuda.empty_cache()
                else:
                    scores = compute_routing_scores_vectorized(inp_flat, U, normalize_k=normalize_k)  # (n_tokens, n_edits)
                    
                    # Get pre-loaded t and tau (GPU if preloaded, else moved on-demand)
                    t = self.t_stacked[layer]  # (n_edits,)
                    tau = self.tau_stacked[layer]  # (n_edits,)
                    if not self.preload_gpu and t.device.type != "cuda":
                        t = t.to(self.device, non_blocking=True)
                        tau = tau.to(self.device, non_blocking=True)
                    
                    # Compute gates: sigmoid((scores - t) / tau)
                    t_expanded = t.unsqueeze(0)  # (1, n_edits)
                    tau_expanded = tau.unsqueeze(0)  # (1, n_edits)
                    gates = torch.sigmoid((scores - t_expanded) / tau_expanded)  # (n_tokens, n_edits)
                    
                    # Apply top-m if enabled
                    if self.hparams.router_top_m > 0 and self.hparams.router_top_m < n_edits:
                        _, top_indices = torch.topk(gates, k=self.hparams.router_top_m, dim=-1)  # (n_tokens, top_m)
                        mask = torch.zeros_like(gates)
                        mask.scatter_(-1, top_indices, 1.0)
                        gates = gates * mask
                    
                    # PERFORMANCE-OPTIMIZED: Compute all deltas at once, then weighted sum
                    # B_stacked: (n_edits, d1, r), Ak: (r, n_tokens)
                    # einsum: 'edr,rt->etd' -> (n_edits, n_tokens, d1)
                    all_deltas = torch.einsum('edr,rt->etd', B_stacked, Ak)  # (n_edits, n_tokens, d1)
                    
                    # Weighted sum: gates (n_tokens, n_edits), all_deltas (n_edits, n_tokens, d1)
                    # einsum: 'te,etd->td' -> (n_tokens, d1)
                    delta_output = torch.einsum('te,etd->td', gates, all_deltas)
            
            else:
                return output
            
            # Reshape and add to output
            delta_output = delta_output.view(batch_size, seq_len, d1)
            
            # Match output shape (handle transposed weights)
            if delta_output.shape[-1] != output.shape[-1]:
                print(f"Warning: shape mismatch in hook. delta: {delta_output.shape}, output: {output.shape}")
                return output
            
            result = output + delta_output
            if not self.preload_gpu and not self.use_chunking:
                # Free temporary GPU copies to cap VRAM usage per layer.
                # (chunking mode doesn't use B_stacked/U_stacked)
                if 'B_stacked' in locals() and B_stacked is not None:
                    del B_stacked
                if 'U' in locals() and U is not None:
                    del U
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            return result
        
        return hook_fn
    
    def apply_hooks(self):
        """Register forward hooks on all target layers."""
        for layer in self.hparams.layers:
            if self.n_edits.get(layer, 0) == 0:
                continue
            module_name = self.hparams.rewrite_module_tmp.format(layer)
            module = nethook.get_module(self.model, module_name)
            hook = module.register_forward_hook(self._make_hook(layer))
            self.hooks.append(hook)
            print(f"[RoutingLoRAHooks] Hook registered for {module_name}")
        
        print(f"[RoutingLoRAHooks] Total {len(self.hooks)} hooks registered (mode={self.hparams.routing_mode})")
        return self
    
    def remove_hooks(self):
        """Remove all registered hooks and free GPU memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Free GPU memory by deleting tensors
        if hasattr(self, 'A_matrices'):
            for layer in list(self.A_matrices.keys()):
                del self.A_matrices[layer]
            self.A_matrices.clear()
        
        if hasattr(self, 'B_stacked'):
            for layer in list(self.B_stacked.keys()):
                del self.B_stacked[layer]
            self.B_stacked.clear()
        
        if hasattr(self, 'U_stacked'):
            for layer in list(self.U_stacked.keys()):
                del self.U_stacked[layer]
            self.U_stacked.clear()
        
        if hasattr(self, 't_stacked'):
            for layer in list(self.t_stacked.keys()):
                del self.t_stacked[layer]
            self.t_stacked.clear()
        
        if hasattr(self, 'tau_stacked'):
            for layer in list(self.tau_stacked.keys()):
                del self.tau_stacked[layer]
            self.tau_stacked.clear()
        
        # Clear CUDA cache to free memory
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        print(f"[RoutingLoRAHooks] All hooks removed and GPU memory freed")
    
    def __enter__(self):
        """Context manager entry."""
        self.apply_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove_hooks()
        return False


def apply_routing_hooks(
    model: AutoModelForCausalLM,
    hparams: MultiLoRAHyperParams,
    A_matrices: Dict[int, torch.Tensor],
    B_bank: Dict[int, List[torch.Tensor]],
    U_bank: Dict[int, List[torch.Tensor]],
    router_stats_bank: Optional[Dict[int, List[Dict]]] = None,  # NEW: for CW-Edit
) -> RoutingLoRAHooks:
    """
    Apply routing hooks to the model for inference.
    
    Usage:
        hooks = apply_routing_hooks(model, hparams, A_matrices, B_bank, U_bank, router_stats_bank)
        # ... do inference ...
        hooks.remove_hooks()
    
    Or as context manager:
        with apply_routing_hooks(model, hparams, A_matrices, B_bank, U_bank, router_stats_bank):
            # ... do inference ...
    
    Returns:
        RoutingLoRAHooks object with applied hooks
    """
    hooks = RoutingLoRAHooks(model, hparams, A_matrices, B_bank, U_bank, router_stats_bank)
    hooks.apply_hooks()
    return hooks
