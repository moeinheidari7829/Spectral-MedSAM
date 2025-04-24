from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x, scale=None):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        if scale is not None:
            new_q = new_q * scale
            new_v = new_v * scale
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class PromptConditioner(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        # Small CNN for richer prompt embedding
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(16, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, C)
        features = self.cnn(x).flatten(1)
        return self.fc(features)


class PromptCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, prompt_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.prompt_proj = nn.Linear(prompt_dim, embed_dim)

    def forward(self, img_tokens, prompt_tokens):
        # img_tokens: (B, N, C), prompt_tokens: (B, M, prompt_dim)
        prompt_emb = self.prompt_proj(prompt_tokens)
        out, _ = self.cross_attn(img_tokens, prompt_emb, prompt_emb)
        return img_tokens + out  # residual


class PromptEncoder(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_prompt_tokens=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.num_prompt_tokens = num_prompt_tokens
        self.embed_dim = embed_dim
        self.proj = nn.Linear(16 * patch_size * patch_size, embed_dim * num_prompt_tokens)

    def forward(self, x):
        # x: (B, 1, PATCH, PATCH)
        B = x.shape[0]
        features = self.cnn(x)  # [B, 16, PATCH, PATCH]
        flat = features.flatten(1)  # [B, 16*PATCH*PATCH]
        tokens = self.proj(flat).view(B, self.num_prompt_tokens, self.embed_dim)
        return tokens


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.
        use_spectral_prompt: if True, expects 4-channel input (image+spectral mask)
        prompt_conditioning: if True, use spectral mask to modulate LoRA layers
        prompt_cross_attention: if True, use cross-attention with prompt tokens after patch embedding
        prompt_multi_scale: if True, inject cross-attention with prompt tokens after every transformer block
    """

    def __init__(self, sam_model: Sam, r: int, lora_layer=None, use_spectral_prompt=False, prompt_conditioning=False,
                 prompt_cross_attention=False, prompt_multi_scale=False, img_size=512, patch_size=16, embed_dim=768, num_heads=12,
                 num_prompt_tokens=8):
        super(LoRA_Sam, self).__init__()
        self.sam = sam_model
        self.use_spectral_prompt = use_spectral_prompt
        self.prompt_conditioning = prompt_conditioning
        self.prompt_cross_attention = prompt_cross_attention
        self.prompt_multi_scale = prompt_multi_scale
        
        # If using spectral prompt, modify the patch embedding to accept 4 channels instead of 3
        if use_spectral_prompt:
            # Save the original weights
            original_proj = self.sam.image_encoder.patch_embed.proj
            in_channels = original_proj.in_channels
            out_channels = original_proj.out_channels
            kernel_size = original_proj.kernel_size
            stride = original_proj.stride
            padding = original_proj.padding
            
            # Create a new projection layer with 4 input channels
            new_proj = nn.Conv2d(4, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            
            with torch.no_grad():
                # Handle both cases: original weights are for 3 or 4 channels
                if original_proj.weight.shape[1] == 3:
                    # Copy the original weights for the first 3 channels
                    new_proj.weight[:, :3, :, :] = original_proj.weight.clone()
                    # Initialize the 4th channel with the average of the RGB channels
                    new_proj.weight[:, 3:4, :, :] = original_proj.weight.mean(dim=1, keepdim=True)
                elif original_proj.weight.shape[1] == 4:
                    # Already 4 channels, just copy
                    new_proj.weight[:, :, :, :] = original_proj.weight.clone()
                else:
                    raise ValueError(f"Unexpected number of input channels in original patch embedding: {original_proj.weight.shape[1]}")
                if original_proj.bias is not None:
                    new_proj.bias = nn.Parameter(original_proj.bias.clone())
            
            # Replace the projection layer
            self.sam.image_encoder.patch_embed.proj = new_proj
            # Always initialize prompt_encoder for spectral prompt mode
            self.prompt_encoder = PromptEncoder(in_channels=1, patch_size=img_size // patch_size, embed_dim=embed_dim,
                                                num_prompt_tokens=num_prompt_tokens)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_prompt_tokens = num_prompt_tokens
        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))
        self.w_As = []
        self.w_Bs = []
        self.conditioners = nn.ModuleList() if prompt_conditioning else None
        if prompt_cross_attention or prompt_multi_scale:
            self.prompt_encoder = PromptEncoder(in_channels=1, patch_size=img_size // patch_size, embed_dim=embed_dim,
                                                num_prompt_tokens=num_prompt_tokens)
        if prompt_cross_attention:
            self.cross_attn = PromptCrossAttention(embed_dim, num_heads, embed_dim)
        if prompt_multi_scale:
            n_blocks = len(sam_model.image_encoder.blocks)
            self.cross_attn_multi = nn.ModuleList([
                PromptCrossAttention(embed_dim, num_heads, embed_dim) for _ in range(n_blocks)
            ])

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
            if prompt_conditioning:
                self.conditioners.append(PromptConditioner(in_channels=1, embed_dim=self.dim))
        self.reset_parameters()

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size, spectral_mask=None, use_cross_attention=False):
        import torch.nn.functional as F
        # --- Multi-Scale Prompt Cross-Attention ---
        if self.prompt_multi_scale and spectral_mask is not None:
            patch_embed = self.sam.image_encoder.patch_embed
            img_tokens = patch_embed(batched_input)  # (B, embed_dim, H', W')
            B, C, H, W = img_tokens.shape
            img_tokens_flat = img_tokens.flatten(2).transpose(1, 2)  # (B, N, C)
            # Downsample spectral_mask to patch size
            patch_hw = self.img_size // self.patch_size
            if spectral_mask.shape[-2:] != (patch_hw, patch_hw):
                spectral_mask_ds = F.interpolate(
                    spectral_mask.unsqueeze(1) if spectral_mask.dim() == 3 else spectral_mask,
                    size=(patch_hw, patch_hw), mode='bilinear', align_corners=False
                )
                if spectral_mask_ds.shape[1] != 1:
                    spectral_mask_ds = spectral_mask_ds[:, 0:1]
            else:
                spectral_mask_ds = spectral_mask.unsqueeze(1) if spectral_mask.dim() == 3 else spectral_mask
            prompt_tokens = self.prompt_encoder(spectral_mask_ds)
            # Pass through transformer blocks, injecting cross-attn after each
            x = img_tokens_flat
            for i, blk in enumerate(self.sam.image_encoder.blocks):
                x = blk(x)
                x = self.cross_attn_multi[i](x, prompt_tokens)
            # Reshape back
            x = x.transpose(1, 2).view(B, C, H, W)
            # Replace patch embedding output
            self.sam.image_encoder.patch_embed = lambda _: x
            return self.sam(batched_input, multimask_output, image_size)
        # --- Prompt Cross-Attention (single injection) ---
        if (self.prompt_cross_attention or use_cross_attention) and spectral_mask is not None:
            patch_embed = self.sam.image_encoder.patch_embed
            img_tokens = patch_embed(batched_input)  # (B, embed_dim, H', W')
            B, C, H, W = img_tokens.shape
            img_tokens_flat = img_tokens.flatten(2).transpose(1, 2)  # (B, N, C)
            # Downsample spectral_mask to patch size
            patch_hw = self.img_size // self.patch_size
            if spectral_mask.shape[-2:] != (patch_hw, patch_hw):
                spectral_mask_ds = F.interpolate(
                    spectral_mask.unsqueeze(1) if spectral_mask.dim() == 3 else spectral_mask,
                    size=(patch_hw, patch_hw), mode='bilinear', align_corners=False
                )
                if spectral_mask_ds.shape[1] != 1:
                    spectral_mask_ds = spectral_mask_ds[:, 0:1]
            else:
                spectral_mask_ds = spectral_mask.unsqueeze(1) if spectral_mask.dim() == 3 else spectral_mask
            prompt_tokens = self.prompt_encoder(spectral_mask_ds)
            img_tokens_attn = self.cross_attn(img_tokens_flat, prompt_tokens)
            img_tokens_attn = img_tokens_attn.transpose(1, 2).view(B, C, H, W)
            # Save the original patch_embed
            original_patch_embed = self.sam.image_encoder.patch_embed
            # Replace patch embedding output with our processed tokens
            self.sam.image_encoder.patch_embed = lambda x: img_tokens_attn
            result = self.sam(batched_input, multimask_output, image_size)
            # Restore the original patch_embed
            self.sam.image_encoder.patch_embed = original_patch_embed
            return result
        # --- Prompt Dropout for conditioning ---
        if self.prompt_conditioning and self.training and spectral_mask is not None:
            if torch.rand(1).item() < 0.2:
                spectral_mask = torch.zeros_like(spectral_mask)
        if self.prompt_conditioning and spectral_mask is not None:
            scales = [cond(spectral_mask) for cond in self.conditioners]
            for i, blk in enumerate(self.sam.image_encoder.blocks):
                if i in self.lora_layer:
                    blk.attn.qkv._current_scale = scales[i]
            return self.sam(batched_input, multimask_output, image_size)
        elif self.use_spectral_prompt and spectral_mask is not None:
            if batched_input.shape[1] == 3:
                batched_input = torch.cat([batched_input, spectral_mask.unsqueeze(1) if spectral_mask.ndim == 3 else spectral_mask], dim=1)
            # Downsample spectral_mask for prompt_encoder
            patch_hw = self.img_size // self.patch_size
            if spectral_mask.shape[-2:] != (patch_hw, patch_hw):
                spectral_mask_ds = F.interpolate(
                    spectral_mask.unsqueeze(1) if spectral_mask.dim() == 3 else spectral_mask,
                    size=(patch_hw, patch_hw), mode='bilinear', align_corners=False
                )
                if spectral_mask_ds.shape[1] != 1:
                    spectral_mask_ds = spectral_mask_ds[:, 0:1]
            else:
                spectral_mask_ds = spectral_mask.unsqueeze(1) if spectral_mask.dim() == 3 else spectral_mask
            _ = self.prompt_encoder(spectral_mask_ds)  # If you need prompt tokens, assign to variable
            return self.sam(batched_input, multimask_output, image_size)
        else:
            return self.sam(batched_input, multimask_output, image_size)


if __name__ == "__main__":
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    lora_sam = LoRA_Sam(sam, 4)
    lora_sam.sam.image_encoder(torch.rand(size=(1, 3, 1024, 1024)))
