
# fm_attack_wrapper.py
# A template wrapper that makes CLIP-style 14-label CXR evaluation plug-and-play for gradient-based attacks
# (PGD / FGSM / DeepFool / CW etc.).
#
# Expected usage:
#   wrapper = FMAttackWrapper(clip_model, device="cuda")
#   logits = wrapper.forward_logits(x)              # [B, 14] (logit for positive vs negative prompt)
#   loss   = wrapper.loss_fn(logits, y_true)        # scalar
#
# Notes:
# - This wrapper assumes you are using the *pair-prompt* zero-shot setting:
#     positive prompt: "{}"
#     negative prompt: "no {}"
#   and then converting to probability by softmax over the pair, i.e.:
#     p_pos = exp(pos) / (exp(pos) + exp(neg))
#   which is equivalent to:
#     logit = pos - neg,  p_pos = sigmoid(logit)
#
# - It also supports pixel-domain clamping and epsilon conversion utilities.

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Your project already uses OpenAI-CLIP style tokenize in zero_shot.py.
    import clip  # type: ignore
except Exception as e:
    clip = None  # the user can still inject their own tokenizer


Tensor = torch.Tensor


@dataclass(frozen=True)
class InputDomain:
    """Defines what 'x' means when attacks call wrapper.forward_logits(x)."""
    # The numeric range of x before preprocessing (clamping is done in that range).
    value_range: Tuple[float, float] = (0.0, 255.0)  # default: pixel domain like your CXR pipeline
    # Whether x is already normalized or not.
    space: Literal["pixel", "unit", "normalized"] = "pixel"
    # Channel convention expected by wrapper.forward_logits
    layout: Literal["NCHW"] = "NCHW"


class FMAttackWrapper(nn.Module):
    """A minimal FM wrapper exposing:
    - forward_logits(x) -> logits suitable for attacks (pre-sigmoid / pre-softmax)
    - loss_fn(logits, y) -> scalar differentiable loss
    - input-domain & epsilon conversion helpers

    This version targets the CLIP-style CXR 14-label 'pair prompt' evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        class_names: Sequence[str],
        pos_template: str = "{}",
        neg_template: str = "no {}",
        context_length: int = 77,
        device: Union[str, torch.device] = "cpu",
        # --- Image preprocessing (match your CLIP-preprocess choices) ---
        input_resolution: Optional[int] = 320,
        resize_mode: Literal["none", "bilinear", "bicubic"] = "none",
        # mean/std in *the same numeric scale as x before Normalize*
        # (your current pipeline normalizes in pixel scale, so defaults are pixel-scale mean/std)
        mean: Tuple[float, float, float] = (101.48761, 101.48761, 101.48761),
        std: Tuple[float, float, float] = (83.43944, 83.43944, 83.43944),
        input_domain: InputDomain = InputDomain(),
        clamp: bool = True,
        tokenizer = None,
    ):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.class_names = list(class_names)
        self.num_labels = len(self.class_names)

        self.pos_template = pos_template
        self.neg_template = neg_template
        self.context_length = context_length

        self.input_resolution = input_resolution
        self.resize_mode = resize_mode

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        self.input_domain = input_domain
        self.clamp = clamp

        # Tokenizer: default to clip.tokenize if available, else user must pass one
        if tokenizer is None:
            if clip is None:
                raise RuntimeError(
                    "clip is not importable. Pass a tokenizer callable that maps List[str] -> LongTensor."
                )
            tokenizer = lambda texts: clip.tokenize(texts, context_length=self.context_length)
        self._tokenizer = tokenizer

        # Precompute text features for pos/neg prompts.
        # Stored as buffers so attacks can backprop only through image path.
        self._build_text_features()

    @torch.no_grad()
    def _build_text_features(self) -> None:
        self.model.eval()

        pos_texts = [self.pos_template.format(c) for c in self.class_names]
        neg_texts = [self.neg_template.format(c) for c in self.class_names]

        pos_tokens = self._tokenizer(pos_texts).to(self.device)
        neg_tokens = self._tokenizer(neg_texts).to(self.device)

        if not hasattr(self.model, "encode_text"):
            raise AttributeError("Model must expose encode_text().")
        pos_feat = self.model.encode_text(pos_tokens).float()
        neg_feat = self.model.encode_text(neg_tokens).float()

        pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
        neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)

        self.register_buffer("pos_text_features", pos_feat)  # [L, D]
        self.register_buffer("neg_text_features", neg_feat)  # [L, D]

    def preprocess(self, x: Tensor) -> Tensor:
        """Convert x to the model's expected input tensor (NCHW, float32, normalized).
        Expected x layout: NCHW.
        Default assumes pixel domain [0,255].
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor NCHW, got shape={tuple(x.shape)}")

        x = x.to(self.device).float()

        # If grayscale (C=1), repeat to 3 channels (CLIP expects 3 channels).
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected C=1 or C=3, got C={x.shape[1]}")        

        # Optional clamp in the *declared input domain*
        if self.clamp and self.input_domain.space in ("pixel", "unit"):
            lo, hi = self.input_domain.value_range
            x = x.clamp(lo, hi)

        # Optional resize (use with care: resizing changes the meaning of epsilon)
        if self.input_resolution is not None and self.resize_mode != "none":
            mode = "bilinear" if self.resize_mode == "bilinear" else "bicubic"
            x = F.interpolate(x, size=(self.input_resolution, self.input_resolution), mode=mode, align_corners=False)

        # Normalize (pixel-scale by default)
        x = (x - self.mean) / self.std
        return x

    def forward_logits(self, x: Tensor) -> Tensor:
        """Return per-label logits in a *binary* form: logit = score(pos_prompt) - score(neg_prompt).
        Shape: [B, L]. These are suitable for BCEWithLogitsLoss and for gradient-based attacks.
        """
        x_in = self.preprocess(x)

        if not hasattr(self.model, "encode_image"):
            raise AttributeError("Model must expose encode_image().")
        image_feat = self.model.encode_image(x_in).float()
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)  # [B, D]

        # Pair logits: [B, L]
        pos = image_feat @ self.pos_text_features.t()
        neg = image_feat @ self.neg_text_features.t()

        # Equivalent to softmax([pos, neg]) on the pair per label:
        # p_pos = exp(pos)/(exp(pos)+exp(neg)) = sigmoid(pos-neg).
        logits = pos - neg
        return logits

    @staticmethod
    def loss_fn(
        logits: Tensor,
        y: Tensor,
        *,
        reduction: Literal["mean", "sum"] = "mean",
        pos_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Binary multi-label loss for 14-label CXR.
        - logits: [B, L] (pre-sigmoid)
        - y:      [B, L] in {0,1} (float or bool)
        Returns: scalar loss.
        """
        if y.dtype != torch.float32:
            y = y.float()
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction=reduction, pos_weight=pos_weight)
        return loss

    # -------------------------
    # Epsilon conversion helpers
    # -------------------------

    def eps_pixel_to_unit(self, eps_pixel: float) -> float:
        """Convert epsilon from [0,255] pixel domain to [0,1] domain."""
        return float(eps_pixel) / 255.0

    def eps_unit_to_pixel(self, eps_unit: float) -> float:
        """Convert epsilon from [0,1] domain to [0,255] pixel domain."""
        return float(eps_unit) * 255.0

    def eps_pixel_to_normalized(self, eps_pixel: float) -> Tensor:
        """Convert L∞ epsilon in pixel domain to the equivalent per-channel bound in normalized space:
        x_norm = (x - mean)/std  =>  δ_norm = δ / std.
        Returns a length-3 tensor [eps_R, eps_G, eps_B] in normalized space.
        """
        eps = torch.tensor([eps_pixel, eps_pixel, eps_pixel], device=self.device, dtype=torch.float32)
        return eps / self.std.view(-1)

    def eps_unit_to_normalized(self, eps_unit: float, *, assume_unit_mean_std: bool = False) -> Tensor:
        """Convert L∞ epsilon in [0,1] domain to normalized space.
        If you normalize in unit space as x_norm=(x-mean01)/std01, then δ_norm=δ01/std01.
        This project currently normalizes in pixel space, so by default we:
            eps_pixel = eps_unit * 255
            eps_norm  = eps_pixel / std_pixel
        Set assume_unit_mean_std=True only if you *actually* normalize in [0,1] space with std01.
        """
        if assume_unit_mean_std:
            raise NotImplementedError(
                "Pass your unit-space std01 to compute eps_norm = eps_unit/std01."
            )
        return self.eps_pixel_to_normalized(self.eps_unit_to_pixel(eps_unit))
