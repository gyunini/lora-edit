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
) -> torch.Tensor:
    """
    Compute routing scores with vectorized operations (FAST).
    
    PDF Eq.21:
    s_i(k) = ||U_i.T @ k||²
    
    Args:
        k: Input key vector, shape (batch, d0)
        U_stacked: Stacked subspace bases, shape (n_edits, d0, r_route)
    
    Returns:
        scores: Routing scores, shape (batch, n_edits)
    """
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
) -> torch.Tensor:
    """
    Compute routing scores for input key k (legacy interface).
    
    PDF Eq.21:
    s_i(k) = ||U_i.T @ k||² = k.T @ P_i @ k
    
    Args:
        k: Input key vector, shape (d0,) or (batch, d0)
        U_bank: List of subspace bases U_j, each shape (d0 × r_j)
    
    Returns:
        scores: Routing scores, shape (n_edits,) or (batch, n_edits)
    """
    if len(U_bank) == 0:
        return torch.tensor([])
    
    # Handle batched input
    is_batched = k.dim() == 2
    if not is_batched:
        k = k.unsqueeze(0)  # (1, d0)
    
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


def route_and_aggregate(
    k: torch.Tensor,
    B_bank: List[torch.Tensor],
    U_bank: List[torch.Tensor],
    A: torch.Tensor,
    routing_mode: str = "soft",
    gamma: float = 10.0,
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
        routing_mode: "none", "hard", or "soft"
        gamma: Temperature for soft routing
    
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
    
    else:
        raise ValueError(f"Unknown routing_mode: {routing_mode}")
    
    return delta


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
) -> Tuple[AutoModelForCausalLM, Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]]]:
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
    
    Returns:
        model: Updated model
        B_bank: Updated B_bank with new B_j matrices
        U_bank: Updated U_bank with new U_j matrices
    """
    # Initialize U_bank if not provided
    if U_bank is None:
        U_bank = {layer: [] for layer in hparams.layers}

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
    return model, B_bank, U_bank


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
        device: str = "cuda",
    ):
        self.model = model
        self.hparams = hparams
        self.hooks = []
        self.device = device
        
        # ===== Routing tensor placement =====
        self.preload_gpu = getattr(hparams, "router_preload_gpu", True)
        if self.preload_gpu:
            print("[RoutingLoRAHooks] Setting up performance-optimized routing (pre-loading to GPU)...")
        else:
            print("[RoutingLoRAHooks] Setting up memory-efficient routing (CPU-stacked, GPU on-demand)...")
        
        self.A_matrices = {}     # {layer: A (r, d0) on GPU}
        self.B_stacked = {}      # {layer: (n_edits, d1, r) stacked (GPU if preloaded, else CPU)}
        self.U_stacked = {}      # {layer: (n_edits, d0, r_route) stacked (GPU if preloaded, else CPU)}
        self.n_edits = {}        # {layer: n_edits}
        
        for layer in hparams.layers:
            n = len(B_bank[layer])
            self.n_edits[layer] = n
            
            if n == 0:
                continue
            
            # Pre-load A to GPU (used in every forward pass)
            self.A_matrices[layer] = A_matrices[layer].to(device)  # (r, d0) on GPU
            
            # Stack B_j for vectorized operations (GPU if preloaded, else CPU)
            if self.preload_gpu:
                B_list = [B.to(device) for B in B_bank[layer]]
            else:
                B_list = [B.cpu() for B in B_bank[layer]]
            self.B_stacked[layer] = torch.stack(B_list, dim=0)  # (n_edits, d1, r)
            
            # Stack U matrices (GPU if preloaded, else CPU)
            if self.preload_gpu:
                U_list = [U.to(device) for U in U_bank[layer]]
            else:
                U_list = [U.cpu() for U in U_bank[layer]]
            self.U_stacked[layer] = torch.stack(U_list, dim=0)  # (n_edits, d0, r_route)
            
            # Memory usage info
            d1 = self.B_stacked[layer].shape[1]
            r = self.A_matrices[layer].shape[0]
            d0 = self.A_matrices[layer].shape[1]
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
            B_stacked = self.B_stacked[layer]
            U = self.U_stacked[layer]
            if not self.preload_gpu and B_stacked.device.type != "cuda":
                B_stacked = B_stacked.to(self.device, non_blocking=True)
                U = U.to(self.device, non_blocking=True)
            n_edits = B_stacked.shape[0]
            d1 = B_stacked.shape[1]
            r = A.shape[0]
            
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
            
            else:
                return output
            
            # Reshape and add to output
            delta_output = delta_output.view(batch_size, seq_len, d1)
            
            # Match output shape (handle transposed weights)
            if delta_output.shape[-1] != output.shape[-1]:
                print(f"Warning: shape mismatch in hook. delta: {delta_output.shape}, output: {output.shape}")
                return output
            
            result = output + delta_output
            if not self.preload_gpu:
                # Free temporary GPU copies to cap VRAM usage per layer.
                del B_stacked, U
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
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print(f"[RoutingLoRAHooks] All hooks removed")
    
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
) -> RoutingLoRAHooks:
    """
    Apply routing hooks to the model for inference.
    
    Usage:
        hooks = apply_routing_hooks(model, hparams, A_matrices, B_bank, U_bank)
        # ... do inference ...
        hooks.remove_hooks()
    
    Or as context manager:
        with apply_routing_hooks(model, hparams, A_matrices, B_bank, U_bank):
            # ... do inference ...
    
    Returns:
        RoutingLoRAHooks object with applied hooks
    """
    hooks = RoutingLoRAHooks(model, hparams, A_matrices, B_bank, U_bank)
    hooks.apply_hooks()
    return hooks
