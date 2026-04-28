import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from scipy.ndimage import binary_dilation, binary_erosion


def _build_norm_3d(num_channels: int, norm: str, data_format: str) -> nn.Module:
    if norm.upper() == 'LN':
        if data_format == 'channels_last':
            return nn.LayerNorm(num_channels)
        else:
            # channels_first -> use GroupNorm with 1 group to mimic LN across C
            return nn.GroupNorm(1, num_channels)
    if norm.upper() == 'BN':
        return nn.BatchNorm3d(num_channels)
    if norm.upper() == 'IN':
        return nn.InstanceNorm3d(num_channels, affine=True)
    return nn.Identity()


def _build_act(act: str) -> nn.Module:
    act = act.upper()
    if act == 'GELU':
        return nn.GELU()
    if act == 'RELU':
        return nn.ReLU(inplace=True)
    if act == 'SILU' or act == 'SWISH':
        return nn.SiLU(inplace=True)
    return nn.Identity()


def _offset_mask_param_groups(model: nn.Module, base_lr: float, weight_decay: float, offset_lr_scale: float = 0.25) -> List[Dict[str, Any]]:
    dcn_params: List[nn.Parameter] = []
    base_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ('offset' in name) or ('mask' in name):
            dcn_params.append(p)
        else:
            base_params.append(p)
    return [
        {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": dcn_params,  "lr": base_lr * offset_lr_scale, "weight_decay": weight_decay},
    ]


def create_adamw_for_better(model: nn.Module, lr: float, weight_decay: float, offset_lr_scale: float = 0.25) -> torch.optim.Optimizer:
    groups = _offset_mask_param_groups(model, lr, weight_decay, offset_lr_scale)
    return torch.optim.AdamW(groups)


def clip_offset_mask_grads(model: nn.Module, max_norm: float = 1.0) -> None:
    params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if ('offset' in name) or ('mask' in name):
            params.append(p)
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


class DCNv3_3D(nn.Module):
    """Implementation detail for the GR-GCDCN model."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
        group: int = 2,
        act_layer: str = 'GELU',
        norm_layer: str = 'LN',
        
        offset_scale_warmup_start: float = 0.5,
        warmup_steps: int = 2000,
        
        offset_axis_scale: Tuple[float, float, float] = (0.5, 1.0, 1.0),
        enable_monitor: bool = False,
        auto_choose_group: bool = True,
        return_offsets: bool = False,
        use_residual_gate: bool = False,
    ) -> None:
        super().__init__()
        if auto_choose_group:
            group = self._choose_group(channels, group)
        assert channels % group == 0, 'See the source code context for details.'
        self.channels = channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group = group
        self.group_channels = channels // group
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.register_buffer(
            'offset_axis_scale_tensor',
            torch.tensor(offset_axis_scale, dtype=torch.float32)
        )
        self.enable_monitor = enable_monitor
        self.return_offsets = return_offsets
        self.use_residual_gate = use_residual_gate

        
        self.offset_scale_warmup_start = float(offset_scale_warmup_start)
        self.warmup_steps = int(warmup_steps)
        self.register_buffer('global_step', torch.zeros((), dtype=torch.long))

        
        self.dw_conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, groups=channels, bias=False),
            _build_norm_3d(channels, norm_layer, 'channels_first'),
            _build_act(act_layer),
        )

        
        self.input_proj = nn.Linear(channels, channels)
        P = kernel_size ** 3
        self.offset = nn.Linear(channels, group * P * 3)
        self.mask = nn.Linear(channels, group * P)
        self.output_proj = nn.Linear(channels, out_channels)

        self._reset_parameters(P)
        self._last_offset_stats: Optional[Dict[str, float]] = None
        self._last_sampling_points: Optional[torch.Tensor] = None  # (P,3) in [-1,1] as (z,y,x)
        self._last_input_sizes: Optional[Tuple[int, int, int]] = None  # (D_p,H_p,W_p)
        self._last_mask: Optional[torch.Tensor] = None
        self._last_offsets: Optional[torch.Tensor] = None
        if self.use_residual_gate:
            self.gate = nn.Parameter(torch.tensor(0.0))
            self.residual_proj = nn.Conv3d(channels, out_channels, kernel_size=1, stride=self.stride, bias=False)
        else:
            self.register_parameter('gate', None)
            self.residual_proj = None

    @staticmethod
    def _choose_group(channels: int, default_group: int) -> int:
        if channels % default_group == 0:
            return default_group
        if channels % 4 == 0:
            return 4
        if channels % 2 == 0:
            return 2
        return 1

    def _reset_parameters(self, P: int) -> None:
        
        nn.init.constant_(self.offset.weight, 0.0)
        nn.init.constant_(self.offset.bias, 0.0)
        
        
        nn.init.constant_(self.mask.weight, 0.0)
        with torch.no_grad():
            if self.mask.bias is not None:
                g = self.group
                bias = torch.full((g, P), -5.0, dtype=self.mask.bias.dtype)
                center_index = P // 2
                bias[:, center_index] = 0.0  
                self.mask.bias.copy_(bias.reshape(-1))
        
        
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def step(self, n: int = 1) -> None:
        
        self.global_step += int(n)

    def get_current_offset_scale(self) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        s = int(self.global_step.item())
        if s >= self.warmup_steps:
            return 1.0
        alpha = s / float(self.warmup_steps)
        return float(self.offset_scale_warmup_start * (1.0 - alpha) + 1.0 * alpha)

    def _compute_stats(self, off: torch.Tensor) -> None:
        if not self.enable_monitor:
            return
        # off: (N,D,H,W,g,P,3) in input-normalized coords
        with torch.no_grad():
            try:
                m = off.mean().item()
                s = off.std().item()
                mx = off.abs().max().item()
                
                
                if self._last_offset_stats is None or mx > 0 or (abs(m) > 1e-8 or abs(s) > 1e-8):
                    self._last_offset_stats = {"mean": m, "std": s, "max_abs": mx}
            except Exception as e:
                
                print(f"Failed to compute DCN offset statistics: {e}")
                self._last_offset_stats = None

    def get_last_offset_stats(self) -> Optional[Dict[str, float]]:
        return self._last_offset_stats

    @torch.no_grad()
    def get_last_sampling_points(self) -> Optional[torch.Tensor]:
        return self._last_sampling_points

    @torch.no_grad()
    def save_sampling_on_slice(self, save_path: str, axis: str = 'xy', slice_index: Optional[int] = None) -> bool:
        if self._last_sampling_points is None or self._last_input_sizes is None:
            return False
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            return False
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        D_p, H_p, W_p = self._last_input_sizes
        pts = self._last_sampling_points  # (P,3) (z,y,x) in [-1,1]
        x = ((pts[:, 2] + 1) * 0.5) * (W_p - 1)
        y = ((pts[:, 1] + 1) * 0.5) * (H_p - 1)
        z = ((pts[:, 0] + 1) * 0.5) * (D_p - 1)
        axis = axis.lower()
        if axis == 'xy':
            if slice_index is None:
                slice_index = int(round(z.mean().item()))
            xs, ys = x.numpy(), y.numpy()
        elif axis == 'xz':
            if slice_index is None:
                slice_index = int(round(y.mean().item()))
            xs, ys = x.numpy(), z.numpy()
        else:  # yz
            if slice_index is None:
                slice_index = int(round(x.mean().item()))
            xs, ys = y.numpy(), z.numpy()
        plt.figure(figsize=(4, 4))
        plt.scatter(xs, ys, s=8, c='r')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return True

    @staticmethod
    def _build_reference_grid(D_out: int, H_out: int, W_out: int, stride: int, pad: int,
                              D_p: int, H_p: int, W_p: int, device: torch.device) -> torch.Tensor:
        
        z, y, x = torch.meshgrid(
            torch.arange(D_out, device=device, dtype=torch.float32),
            torch.arange(H_out, device=device, dtype=torch.float32),
            torch.arange(W_out, device=device, dtype=torch.float32),
            indexing='ij'
        )
        x = x * stride + pad + 0.5
        y = y * stride + pad + 0.5
        z = z * stride + pad + 0.5
        ref = torch.stack((x / W_p, y / H_p, z / D_p), dim=-1)  # (D_out,H_out,W_out,3)
        return ref.unsqueeze(0).unsqueeze(4)  # (1,D_out,H_out,W_out,1,3)

    @staticmethod
    def _build_kernel_grid(k: int, dilation: int, D_p: int, H_p: int, W_p: int, device: torch.device) -> torch.Tensor:
        kz, ky, kx = torch.meshgrid(
            torch.linspace(-(k - 1) // 2, (k - 1) // 2, k, device=device),
            torch.linspace(-(k - 1) // 2, (k - 1) // 2, k, device=device),
            torch.linspace(-(k - 1) // 2, (k - 1) // 2, k, device=device),
            indexing='ij'
        )
        kx = kx * dilation
        ky = ky * dilation
        kz = kz * dilation
        grid = torch.stack((kx / W_p, ky / H_p, kz / D_p), dim=-1)  # (k,k,k,3)
        P = k * k * k
        return grid.reshape(1, 1, 1, 1, P, 3)  # (1,1,1,1,P,3)

    def _dcn_v3_core(self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (N,D_in,H_in,W_in,C), offset:(N,D_out,H_out,W_out, g*P*3), mask:(N,D_out,H_out,W_out, g*P)
        N, D_in, H_in, W_in, C = x.shape
        k = self.kernel_size
        P = k * k * k
        g = self.group
        s = self.stride
        d = self.dilation
        device = x.device

        
        x_pad = F.pad(x, [0, 0, self.pad, self.pad, self.pad, self.pad, self.pad, self.pad])
        _, D_p, H_p, W_p, _ = x_pad.shape

        
        D_out = (D_in + 2 * self.pad - d * (k - 1) - 1) // s + 1
        H_out = (H_in + 2 * self.pad - d * (k - 1) - 1) // s + 1
        W_out = (W_in + 2 * self.pad - d * (k - 1) - 1) // s + 1

        
        ref = self._build_reference_grid(D_out, H_out, W_out, s, self.pad, D_p, H_p, W_p, device)  # (1,Do,Ho,Wo,1,3)
        kgrid = self._build_kernel_grid(k, d, D_p, H_p, W_p, device)  # (1,1,1,1,P,3)

        
        offset = offset.view(N, D_out, H_out, W_out, g, P, 3)
        mask = mask.view(N, D_out, H_out, W_out, g, P)

        
        axis = self.offset_axis_scale_tensor.view(1, 1, 1, 1, 1, 1, 3).to(device)
        off_norm = offset / torch.tensor([W_p, H_p, D_p], device=device).view(1, 1, 1, 1, 1, 1, 3)
        cur_scale = self.get_current_offset_scale()
        
        ref_exp = ref.unsqueeze(5).expand(1, D_out, H_out, W_out, g, P, 3)
        kgrid_exp = kgrid.expand(1, D_out, H_out, W_out, g, P, 3)
        sampling_locs = ref_exp + kgrid_exp + off_norm * axis * float(cur_scale)  # (N,Do,Ho,Wo,g,P,3)
        sampling_locs = sampling_locs * 2.0 - 1.0

        
        if self.enable_monitor:
            self._compute_stats(off_norm)
        if self.enable_monitor:
            try:
                cz, cy, cx = D_out // 2, H_out // 2, W_out // 2
                self._last_sampling_points = sampling_locs[0, cz, cy, cx, 0].detach().cpu()
                self._last_input_sizes = (D_p, H_p, W_p)
            except Exception:
                pass

        
        x_in = x_pad.permute(0, 4, 1, 2, 3).reshape(N * g, self.group_channels, D_p, H_p, W_p)

        
        N_group = N * g
        grids = sampling_locs.permute(0, 4, 5, 1, 2, 3, 6).reshape(N_group * P, D_out, H_out, W_out, 3)
        x_rep = x_in.unsqueeze(1).expand(N_group, P, self.group_channels, D_p, H_p, W_p)
        x_rep = x_rep.reshape(N_group * P, self.group_channels, D_p, H_p, W_p)

        sampled = F.grid_sample(x_rep, grids, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled = sampled.view(N_group, P, self.group_channels, D_out, H_out, W_out)
        sampled = sampled.permute(0, 2, 3, 4, 5, 1)  # (N*g, gc, Do,Ho,Wo, P)

        
        m = mask.permute(0, 4, 1, 2, 3, 5).reshape(N_group, 1, D_out, H_out, W_out, P)
        out = (sampled * m).sum(-1)  # (N*g, gc, Do,Ho,Wo)
        out = out.view(N, g * self.group_channels, D_out, H_out, W_out)
        out = out.permute(0, 2, 3, 4, 1)  # (N,Do,Ho,Wo,C)
        
        return out

    def forward(self, x: torch.Tensor, return_offsets: bool = False):
        # x: (N,C,D,H,W)
        N, C, D, H, W = x.shape
        assert C == self.channels, f"Input channels {C} do not match model channels {self.channels}."
        should_return_offsets = return_offsets or self.return_offsets
        residual = None
        if self.use_residual_gate:
            residual = self.residual_proj(x)

        
        k = self.kernel_size
        s = self.stride
        d = self.dilation
        D_out = (D + 2 * self.pad - d * (k - 1) - 1) // s + 1
        H_out = (H + 2 * self.pad - d * (k - 1) - 1) // s + 1
        W_out = (W + 2 * self.pad - d * (k - 1) - 1) // s + 1
        
        
        assert D_out > 0, f"Output depth {D_out} must be positive for input {D}, stride {s}, dilation {d}, pad {self.pad}, and kernel {k}."
        assert H_out > 0, f"Output height {H_out} must be positive for input {H}, stride {s}, dilation {d}, pad {self.pad}, and kernel {k}."
        assert W_out > 0, f"Output width {W_out} must be positive for input {W}, stride {s}, dilation {d}, pad {self.pad}, and kernel {k}."

        x_cl = x.permute(0, 2, 3, 4, 1)  # (N,D,H,W,C)
        x_proj = self.input_proj(x_cl)

        x_dw = self.dw_conv(x)  # (N,C,D,H,W)
        x_dw_cl = x_dw.permute(0, 2, 3, 4, 1)

        
        x_feat = x_dw_cl.reshape(-1, self.channels)
        P = self.kernel_size ** 3
        g = self.group
        
        
        
        if s > 1 or d > 1:
            
            x_dw_cl_out = F.interpolate(
                x_dw_cl.permute(0, 4, 1, 2, 3),  # (N,C,D,H,W)
                size=(D_out, H_out, W_out),
                mode='trilinear',
                align_corners=False
            ).permute(0, 2, 3, 4, 1)  # (N,D_out,H_out,W_out,C)
            x_feat = x_dw_cl_out.reshape(-1, self.channels)
            offset = self.offset(x_feat).view(N, D_out, H_out, W_out, g * P * 3)
            mask = self.mask(x_feat).view(N, D_out, H_out, W_out, g * P)
            mask = F.softmax(mask.view(N, D_out, H_out, W_out, g, P), dim=-1).view(N, D_out, H_out, W_out, g * P)
        else:
            
            offset = self.offset(x_feat).view(N, D, H, W, g * P * 3)
            mask = self.mask(x_feat).view(N, D, H, W, g * P)
            mask = F.softmax(mask.view(N, D, H, W, g, P), dim=-1).view(N, D, H, W, g * P)

        offset_view = offset.view(N, D_out, H_out, W_out, g, P, 3) if (s > 1 or d > 1) else offset.view(N, D, H, W, g, P, 3)
        mask_view = mask.view(N, D_out, H_out, W_out, g, P) if (s > 1 or d > 1) else mask.view(N, D, H, W, g, P)
        if self.enable_monitor or should_return_offsets:
            self._last_mask = mask_view.detach()
            self._last_offsets = offset_view.detach()

        x_out = self._dcn_v3_core(x_proj, offset, mask)
        x_out = self.output_proj(x_out).permute(0, 4, 1, 2, 3)  # (N,C_out,D,H,W)
        if self.use_residual_gate:
            gate = torch.sigmoid(self.gate)
            if residual is None:
                residual = torch.zeros_like(x_out)
            elif residual.shape[2:] != x_out.shape[2:]:
                residual = F.interpolate(residual, size=x_out.shape[2:], mode='trilinear', align_corners=False)
            x_out = residual + gate * x_out
        if should_return_offsets:
            return x_out, offset_view
        return x_out

    def get_last_mask(self) -> Optional[torch.Tensor]:
        return self._last_mask

    def get_last_offsets(self) -> Optional[torch.Tensor]:
        return self._last_offsets


@torch.no_grad()
def dcn_offset_stats(module: nn.Module) -> Optional[Dict[str, float]]:
    """Implementation detail for the GR-GCDCN model."""
    if isinstance(module, DCNv3_3D):
        return module.get_last_offset_stats()
    
    
    all_stats = []
    dcn_count = 0
    enabled_count = 0
    for m in module.modules():
        if isinstance(m, DCNv3_3D):
            dcn_count += 1
            if m.enable_monitor:
                enabled_count += 1
                stats = m.get_last_offset_stats()
                if stats is not None:
                    all_stats.append(stats)
    
    if not all_stats:
        
        if dcn_count > 0:
            print(f"Warning: found {dcn_count} DCN modules, but no valid offset statistics were available.")
        return None
    
    
    try:
        avg_stats = {
            'mean': sum(s['mean'] for s in all_stats) / len(all_stats),
            'std': sum(s['std'] for s in all_stats) / len(all_stats),
            'max_abs': max(s['max_abs'] for s in all_stats)
        }
        return avg_stats
    except Exception as e:
        print(f"Failed to aggregate DCN offset statistics: {e}")
        return None


@torch.no_grad()
def compute_dcn_stats_from_parameters(module: nn.Module) -> Optional[Dict[str, float]]:
    """Implementation detail for the GR-GCDCN model."""
    all_stats = []
    dcn_count = 0
    
    for m in module.modules():
        if isinstance(m, DCNv3_3D) and m.enable_monitor:
            dcn_count += 1
            
            if hasattr(m, 'offset') and m.offset.weight is not None:
                offset_params = m.offset.weight.data
                mean_val = offset_params.mean().item()
                std_val = offset_params.std().item()
                max_abs_val = offset_params.abs().max().item()
                
                stats = {"mean": mean_val, "std": std_val, "max_abs": max_abs_val}
                all_stats.append(stats)
    
    if not all_stats:
        return None
    
    
    avg_stats = {
        'mean': sum(s['mean'] for s in all_stats) / len(all_stats),
        'std': sum(s['std'] for s in all_stats) / len(all_stats),
        'max_abs': max(s['max_abs'] for s in all_stats)
    }
    return avg_stats



def dcn_step_warmup(module: nn.Module, n: int = 1) -> None:
    if isinstance(module, DCNv3_3D):
        module.step(n)
    for m in module.modules():
        if isinstance(m, DCNv3_3D):
            m.step(n)


def clip_dcn_offset_mask(module: nn.Module, max_norm: float = 1.0) -> None:
    clip_offset_mask_grads(module, max_norm)
