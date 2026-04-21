import os
import copy
import logging
from typing import Dict, Optional, Tuple

import torch

import folder_paths
import comfy.utils
import comfy.sd
from comfy import sdxl_clip


# -----------------------------------------------------------------------------
# Simple caches so we don't dequantize the same files every run.
# -----------------------------------------------------------------------------
_DENSE_SD_CACHE: Dict[Tuple[str, float, str, bool], Dict[str, torch.Tensor]] = {}


# -----------------------------------------------------------------------------
# NVFP4 helpers
# -----------------------------------------------------------------------------
# E2M1 values ignoring sign: 0, 0.5, 1, 1.5, 2, 3, 4, 6
# Use sign bit as the high bit of the nibble.
_FP4_E2M1_LUT = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def _get_clip_path(filename: str) -> str:
    path = folder_paths.get_full_path("clip", filename)
    if path is None:
        raise FileNotFoundError(
            f"Could not resolve '{filename}' in ComfyUI models/clip folder."
        )
    return path


def _get_embedding_dir() -> Optional[str]:
    try:
        emb_dirs = folder_paths.get_folder_paths("embeddings")
        if emb_dirs and len(emb_dirs) > 0:
            return emb_dirs[0]
    except Exception:
        pass
    return None


def _safe_load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state dict from {path}, got {type(sd)}")
    return sd


def _unpack_fp4_bytes(packed: torch.Tensor, nibble_order: str) -> torch.Tensor:
    """
    packed: uint8 tensor [..., packed_cols]
    returns: float32 tensor [..., packed_cols * 2]
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"Expected torch.uint8 packed tensor, got {packed.dtype}")

    packed = packed.cpu()
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    if nibble_order == "lo_hi":
        nibbles = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    elif nibble_order == "hi_lo":
        nibbles = torch.stack((high, low), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    else:
        raise ValueError(f"Unsupported nibble_order={nibble_order}")

    return _FP4_E2M1_LUT[nibbles.long()]


def _sanitize_floating_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    sanitize_stats: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    if not tensor.is_floating_point():
        return tensor

    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return tensor

    bad_count = int((~finite_mask).sum().item())
    if sanitize_stats is None:
        logging.warning(
            "[NVFP4 SDXL Loader] sanitized %s non-finite values in %s",
            bad_count,
            tensor_name,
        )
    else:
        sanitize_stats["tensors"] = int(sanitize_stats.get("tensors", 0)) + 1
        sanitize_stats["values"] = int(sanitize_stats.get("values", 0)) + bad_count
        samples = sanitize_stats.setdefault("samples", [])
        if isinstance(samples, list) and len(samples) < 6:
            samples.append(f"{tensor_name} ({bad_count})")

    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _dequantize_nvfp4_weight(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: Optional[torch.Tensor],
    nibble_order: str,
    apply_global_scale2: bool,
    tensor_name: str,
    sanitize_stats: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    """
    Assumes rowwise storage matching your TE dumps:
      packed_weight: [out_features, in_features / 2] uint8
      weight_scale:  [out_features, in_features / 16] float8_e4m3fn
      weight_scale_2: scalar or broadcastable float32
    """
    if packed_weight.ndim != 2:
        raise ValueError(
            f"Expected 2D packed linear weight, got shape={tuple(packed_weight.shape)}"
        )
    if weight_scale.ndim != 2:
        raise ValueError(
            f"Expected 2D block scale tensor, got shape={tuple(weight_scale.shape)}"
        )

    fp4_vals = _unpack_fp4_bytes(packed_weight, nibble_order=nibble_order).float()
    out_features, in_features = fp4_vals.shape

    block_scales = weight_scale.float().cpu()
    block_scales = _sanitize_floating_tensor(
        block_scales,
        f"{tensor_name}_scale",
        sanitize_stats=sanitize_stats,
    )
    expected_groups = (in_features + 15) // 16

    if block_scales.shape[0] != out_features:
        raise ValueError(
            f"Scale rows mismatch: weight={tuple(packed_weight.shape)} "
            f"scale={tuple(weight_scale.shape)}"
        )

    if block_scales.shape[1] < expected_groups:
        raise ValueError(
            f"Scale cols too small: in_features={in_features}, "
            f"expected_groups={expected_groups}, got scale shape={tuple(weight_scale.shape)}"
        )

    # Some exporters pad scale tensors for alignment; trim if needed.
    if block_scales.shape[1] != expected_groups:
        block_scales = block_scales[:, :expected_groups]

    expanded_scales = block_scales.repeat_interleave(16, dim=1)[:, :in_features]
    dense = fp4_vals * expanded_scales

    if apply_global_scale2 and weight_scale_2 is not None:
        dense = dense * _sanitize_floating_tensor(
            weight_scale_2.float().cpu(),
            f"{tensor_name}_scale_2",
            sanitize_stats=sanitize_stats,
        )

    dense = _sanitize_floating_tensor(
        dense,
        tensor_name,
        sanitize_stats=sanitize_stats,
    )
    return dense.to(torch.float16).contiguous()


def _densify_modelopt_nvfp4_state_dict(
    sd: Dict[str, torch.Tensor],
    nibble_order: str = "lo_hi",
    apply_global_scale2: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Converts ModelOpt-style packed NVFP4 linear weights into dense FP16 tensors,
    while keeping already-dense tensors like embeddings, layer norms, and biases.
    Quantization helper tensors are removed from the final state dict.
    """
    out: Dict[str, torch.Tensor] = {}
    converted = 0
    kept_dense = 0
    skipped_aux = 0
    sanitize_stats: Dict[str, object] = {"tensors": 0, "values": 0, "samples": []}

    aux_suffixes = (
        ".input_scale",
        ".weight_scale",
        ".weight_scale_2",
    )

    for k, v in sd.items():
        if k.endswith(aux_suffixes):
            skipped_aux += 1
            continue

        if (
            k.endswith(".weight")
            and torch.is_tensor(v)
            and v.dtype == torch.uint8
            and (k + "_scale") in sd
        ):
            dense_weight = _dequantize_nvfp4_weight(
                packed_weight=v,
                weight_scale=sd[k + "_scale"],
                weight_scale_2=sd.get(k + "_scale_2"),
                nibble_order=nibble_order,
                apply_global_scale2=apply_global_scale2,
                tensor_name=k,
                sanitize_stats=sanitize_stats,
            )
            out[k] = dense_weight
            converted += 1
            continue

        if torch.is_tensor(v):
            if v.is_floating_point():
                out[k] = _sanitize_floating_tensor(
                    v.to(torch.float16).cpu(),
                    k,
                    sanitize_stats=sanitize_stats,
                ).contiguous()
            else:
                out[k] = v.cpu().contiguous()
            kept_dense += 1
        else:
            out[k] = v
            kept_dense += 1

    logging.info(
        "[NVFP4 SDXL Loader] densify complete: converted=%s kept_dense=%s skipped_aux=%s",
        converted,
        kept_dense,
        skipped_aux,
    )
    if int(sanitize_stats["values"]) > 0:
        sample_text = ", ".join(sanitize_stats.get("samples", []))
        logging.warning(
            "[NVFP4 SDXL Loader] sanitized non-finite values during densify: "
            "tensors=%s values=%s samples=%s",
            sanitize_stats["tensors"],
            sanitize_stats["values"],
            sample_text,
        )
    return out


def _load_dense_state_dict_cached(
    path: str,
    nibble_order: str,
    apply_global_scale2: bool,
) -> Dict[str, torch.Tensor]:
    mtime = os.path.getmtime(path)
    cache_key = (path, mtime, nibble_order, apply_global_scale2)

    if cache_key in _DENSE_SD_CACHE:
        return _DENSE_SD_CACHE[cache_key]

    raw_sd = _safe_load_state_dict(path)
    dense_sd = _densify_modelopt_nvfp4_state_dict(
        raw_sd,
        nibble_order=nibble_order,
        apply_global_scale2=apply_global_scale2,
    )
    _DENSE_SD_CACHE[cache_key] = dense_sd
    return dense_sd


def _count_params(sd: Dict[str, torch.Tensor]) -> int:
    total = 0
    for v in sd.values():
        if torch.is_tensor(v):
            total += v.numel()
    return total


def _identify_sdxl_te_role(sd: Dict[str, torch.Tensor]) -> str:
    q_proj_key = "text_model.encoder.layers.0.self_attn.q_proj.weight"
    deep_g_key = "text_model.encoder.layers.30.mlp.fc1.weight"

    if q_proj_key not in sd:
        raise KeyError(f"Missing expected SDXL q_proj key: {q_proj_key}")

    q_shape = tuple(sd[q_proj_key].shape)

    if deep_g_key in sd:
        if q_shape != (1280, 1280):
            raise ValueError(
                f"Detected clip_g routing key but q_proj shape is wrong: {q_shape}"
            )
        return "clip_g"

    if q_shape != (768, 768):
        raise ValueError(
            f"Detected clip_l candidate but q_proj shape is wrong: {q_shape}"
        )
    return "clip_l"


def _normalize_sdxl_text_encoder_order(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    role_a = _identify_sdxl_te_role(sd_a)
    role_b = _identify_sdxl_te_role(sd_b)

    if role_a == role_b:
        raise ValueError(
            f"Could not form SDXL dual clip pair: got roles {role_a} and {role_b}"
        )

    if role_a == "clip_l" and role_b == "clip_g":
        return sd_a, sd_b

    if role_a == "clip_g" and role_b == "clip_l":
        logging.warning(
            "[NVFP4 SDXL Loader] clip_l/clip_g inputs were reversed; auto-swapping them."
        )
        return sd_b, sd_a

    raise ValueError(f"Unexpected role combination: {role_a}, {role_b}")


class _SDXLTarget:
    def __init__(self):
        self.params = {}
        self.clip = sdxl_clip.SDXLClipModel
        self.tokenizer = sdxl_clip.SDXLTokenizer


# -----------------------------------------------------------------------------
# ComfyUI node
# -----------------------------------------------------------------------------
class NVFP4SDXLDualCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        clip_files = folder_paths.get_filename_list("clip")
        return {
            "required": {
                "clip_l_file": (clip_files,),
                "clip_g_file": (clip_files,),
                "nibble_order": (["lo_hi", "hi_lo"], {"default": "lo_hi"}),
                "apply_global_scale2": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/text encoders"

    def load_clip(
        self,
        clip_l_file: str,
        clip_g_file: str,
        nibble_order: str,
        apply_global_scale2: bool,
    ):
        te1_path = _get_clip_path(clip_l_file)
        te2_path = _get_clip_path(clip_g_file)

        te1_sd = _load_dense_state_dict_cached(
            te1_path,
            nibble_order=nibble_order,
            apply_global_scale2=apply_global_scale2,
        )
        te2_sd = _load_dense_state_dict_cached(
            te2_path,
            nibble_order=nibble_order,
            apply_global_scale2=apply_global_scale2,
        )

        te1_sd, te2_sd = _normalize_sdxl_text_encoder_order(te1_sd, te2_sd)

        total_params = _count_params(te1_sd) + _count_params(te2_sd)
        embedding_dir = _get_embedding_dir()
        target = _SDXLTarget()

        clip = comfy.sd.CLIP(
            target=target,
            embedding_directory=embedding_dir,
            no_init=False,
            tokenizer_data={},
            parameters=total_params,
            state_dict=[te1_sd, te2_sd],
            model_options={},
            disable_dynamic=False,
        )

        return (clip,)

    @classmethod
    def IS_CHANGED(
        cls,
        clip_l_file: str,
        clip_g_file: str,
        nibble_order: str,
        apply_global_scale2: bool,
    ):
        try:
            p1 = _get_clip_path(clip_l_file)
            p2 = _get_clip_path(clip_g_file)
            return f"{p1}:{os.path.getmtime(p1)}|{p2}:{os.path.getmtime(p2)}|{nibble_order}|{apply_global_scale2}"
        except Exception:
            return float("nan")


NODE_CLASS_MAPPINGS = {
    "NVFP4SDXLDualCLIPLoader": NVFP4SDXLDualCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NVFP4SDXLDualCLIPLoader": "NVFP4 SDXL Dual CLIP Loader",
}
