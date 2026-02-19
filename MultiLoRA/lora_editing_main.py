import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
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
        A: torch.Tensor of shape (r, d0), where r = hparams.lora_rank
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
    # 기존 AlphaEdit 방식: threshold 미만의 모든 nullspace 방향 사용
    A = U[:, small_singular_indices].T
    
    print(f"Layer {layer}: nullspace dim = {len(small_singular_indices)}, A shape = {A.shape}")
    return A

def apply_lora_editing_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MultiLoRAHyperParams,
    cache_template: Optional[str] = None,
    A_matrices: Dict[int, torch.Tensor] = None,  # layer -> A matrix
    B_bank: Dict[int, List[torch.Tensor]] = None,  # layer -> list of B_j matrices
) -> Tuple[AutoModelForCausalLM, Dict[int, List[torch.Tensor]]]:
    """
    Executes the LoRA editing algorithm (BA summation approach).
    
    PDF Eq.9-11:
    - X = A @ K_j           # (r × u)
    - M = X @ X.T + A @ A.T # (r × r)  
    - B_j = E @ X.T @ M⁻¹   # (d1 × r)
    - Δ = B_j @ A           # (d1 × d0)
    
    Args:
        A_matrices: Pre-computed nullspace basis A for each layer
        B_bank: Accumulated B matrices from previous edits
    
    Returns:
        model: Updated model
        B_bank: Updated B_bank with new B_j matrices
    """

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"LoRA request sample: "
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
        # Retrieve k/v pair if already stored in cache
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
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        # layer_ks: (d0 × u) where u = number of edits
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        # E = resid: (d1 × u) - residual to be corrected
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        
        # ===== PDF Eq.9-11: Compute B_j =====
        # A: (r × d0), layer_ks: (d0 × u), resid: (d1 × u)
        A = A_matrices[layer].cuda()
        K = layer_ks  # (d0 × u)
        E = resid.T   # (u × d1) -> will transpose back
        
        # X = A @ K  (r × u)
        X = A @ K
        
        # M = X @ X.T + A @ A.T  (r × r)
        # Adding L2 regularization for numerical stability
        M = X @ X.T + A @ A.T + hparams.L2 * torch.eye(A.shape[0], device="cuda")
        
        # B_j = E @ X.T @ M^{-1}  (d1 × r)
        # E is (d1 × u), X.T is (u × r), M^{-1} is (r × r)
        # Using solve for numerical stability: B_j @ M = E @ X.T
        # -> B_j = (M.T \ (E @ X.T).T).T = solve(M.T, (E @ X.T).T).T
        B_j = torch.linalg.solve(M.T, (resid @ X.T).T).T  # (d1 × r)
        
        # Store B_j in B_bank
        B_bank[layer].append(B_j.cpu())
        
        # Compute update: Δ = B_j @ A  (d1 × d0)
        upd_matrix = B_j @ A
        
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        print(f"B_j shape: {B_j.shape}, A shape: {A.shape}")
        
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
        
        # Clear GPU memory
        del layer_ks, cur_zs, targets, upd_matrix, A, X, M, B_j
        torch.cuda.empty_cache()

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, B_bank


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
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

