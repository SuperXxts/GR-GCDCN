"""Implementation detail for the GR-GCDCN model."""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from .DCNv3_3D import DCNv3_3D, _build_norm_3d, _build_act, _offset_mask_param_groups


class StructureTensor3D(nn.Module):
    """Implementation detail for the GR-GCDCN model."""
    
    _profiling_enabled: bool = False
    _profiling_records: List[float] = []
    _profiling_max_samples: Optional[int] = None

    def __init__(
        self,
        window_size: int = 3,
        eps: float = 1e-6,
        jitter: float = 1e-6,
        use_separable: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.eps = eps
        self.jitter = jitter
        self.use_separable = use_separable
        self._warned = 0  
        
        
        self._init_sobel_kernels()

    @classmethod
    def enable_profiling(cls, enable: bool = True, max_samples: Optional[int] = None):
        cls._profiling_enabled = enable
        cls._profiling_max_samples = max_samples

    @classmethod
    def reset_profiling(cls):
        cls._profiling_records = []

    @classmethod
    def get_profiling_stats(cls) -> Optional[Dict[str, float]]:
        if not cls._profiling_records:
            return None
        data = cls._profiling_records
        count = len(data)
        avg = sum(data) / count
        return {
            "count": count,
            "avg_ms": avg,
            "min_ms": min(data),
            "max_ms": max(data),
        }

    @classmethod
    def profiling_active(cls) -> bool:
        return cls._profiling_enabled
    
    def _init_sobel_kernels(self):
        """Implementation detail for the GR-GCDCN model."""
        if self.use_separable:
            
            s = torch.tensor([1., 2., 1.], dtype=torch.float32)  
            d = torch.tensor([-1., 0., 1.], dtype=torch.float32)  
            
            
            Kx = d[:, None, None] * s[None, :, None] * s[None, None, :]  
            Ky = s[:, None, None] * d[None, :, None] * s[None, None, :]  
            Kz = s[:, None, None] * s[None, :, None] * d[None, None, :]  
            
            norm = 8.0
        else:
            
            Kx = torch.tensor([
                [[-1., -3., -1.],
                 [-3., -6., -3.],
                 [-1., -3., -1.]],
                [[ 0.,  0.,  0.],
                 [ 0.,  0.,  0.],
                 [ 0.,  0.,  0.]],
                [[ 1.,  3.,  1.],
                 [ 3.,  6.,  3.],
                 [ 1.,  3.,  1.]],
            ], dtype=torch.float32)
            Ky = Kx.permute(1, 0, 2).contiguous()
            Kz = Kx.permute(2, 1, 0).contiguous()
            norm = 32.0
        
        self.register_buffer('sobel_x_3d', Kx.view(1, 1, 3, 3, 3) / norm)
        self.register_buffer('sobel_y_3d', Ky.view(1, 1, 3, 3, 3) / norm)
        self.register_buffer('sobel_z_3d', Kz.view(1, 1, 3, 3, 3) / norm)
    
    @staticmethod
    def _symmetrize(M: torch.Tensor) -> torch.Tensor:
        """Implementation detail for the GR-GCDCN model."""
        return 0.5 * (M + M.transpose(-1, -2))

    
    def compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implementation detail for the GR-GCDCN model."""
        N, C, _, _, _ = x.shape
        dtype, device = x.dtype, x.device
        
        
        w_x = self.sobel_x_3d.to(device=device, dtype=dtype).repeat(C, 1, 1, 1, 1)
        w_y = self.sobel_y_3d.to(device=device, dtype=dtype).repeat(C, 1, 1, 1, 1)
        w_z = self.sobel_z_3d.to(device=device, dtype=dtype).repeat(C, 1, 1, 1, 1)
        
        
        gx = F.conv3d(x, w_x, padding=1, groups=C)
        gy = F.conv3d(x, w_y, padding=1, groups=C)
        gz = F.conv3d(x, w_z, padding=1, groups=C)
        
        return gx, gy, gz
    
    def build_structure_tensor(self, gx: torch.Tensor, gy: torch.Tensor, gz: torch.Tensor) -> torch.Tensor:
        """Implementation detail for the GR-GCDCN model."""
        N, C, D, H, W = gx.shape
        
        
        T_xx = gx * gx
        T_xy = gx * gy
        T_xz = gx * gz
        T_yy = gy * gy
        T_yz = gy * gz
        T_zz = gz * gz
        
        
        
        if self.window_size > 1 and min(D, H, W) >= self.window_size:
            
            pad = self.window_size // 2
            T_xx = F.avg_pool3d(T_xx, kernel_size=self.window_size, stride=1, padding=pad)
            T_xy = F.avg_pool3d(T_xy, kernel_size=self.window_size, stride=1, padding=pad)
            T_xz = F.avg_pool3d(T_xz, kernel_size=self.window_size, stride=1, padding=pad)
            T_yy = F.avg_pool3d(T_yy, kernel_size=self.window_size, stride=1, padding=pad)
            T_yz = F.avg_pool3d(T_yz, kernel_size=self.window_size, stride=1, padding=pad)
            T_zz = F.avg_pool3d(T_zz, kernel_size=self.window_size, stride=1, padding=pad)
        
        
        T = torch.zeros(N, C, D, H, W, 3, 3, device=gx.device, dtype=gx.dtype)
        
        T[..., 0, 0] = T_xx  # T_xx
        T[..., 0, 1] = T_xy  # T_xy
        T[..., 0, 2] = T_xz  # T_xz
        T[..., 1, 0] = T_xy  # T_yx = T_xy
        T[..., 1, 1] = T_yy  # T_yy
        T[..., 1, 2] = T_yz  # T_yz
        T[..., 2, 0] = T_xz  # T_zx = T_xz
        T[..., 2, 1] = T_yz  # T_zy = T_yz
        T[..., 2, 2] = T_zz  # T_zz
        
        return T
    
    def extract_principal_direction_with_eigenvalues(self, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation detail for the GR-GCDCN model."""
        
        if T.dim() == 7:  # (N, C, D, H, W, 3, 3)
            N, C, D, H, W = T.shape[:5]
            T_flat = T.view(-1, 3, 3)  # (N*C*D*H*W, 3, 3)
            output_shape = (N, C, D, H, W)
        else:  
            N, D, H, W = T.shape[:4]
            T_flat = T.view(-1, 3, 3)  # (N*D*H*W, 3, 3)
            output_shape = (N, D, H, W)
        
        
        
        T = self._symmetrize(T)
        T_flat = T.view(-1, 3, 3)
        B = T_flat.shape[0]

        
        original_dtype = T_flat.dtype
        T32 = T_flat.to(torch.float32)

        
        I = torch.eye(3, dtype=torch.float32, device=T32.device).unsqueeze(0).expand_as(T32)
        trace = T32.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) / 3.0  # (B,1)
        eps = trace * 1e-3 + 1e-6  
        
        T_reg = 0.5 * (T32 + T32.transpose(-1, -2)) + eps.unsqueeze(-1) * I
        
        T_reg = torch.where(torch.isfinite(T_reg), T_reg, torch.zeros_like(T_reg))

        
        
        gradient_strength = T32.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B,)
        low_gradient_mask = gradient_strength <= 1e-6

        try:
            
            with torch.cuda.amp.autocast(enabled=False):  
                eigenvalues, eigenvectors = torch.linalg.eigh(T_reg)  # (B,3), (B,3,3)
            
            
            evals_sorted, idx = torch.sort(eigenvalues, dim=-1, descending=True)
            evecs_sorted = torch.gather(eigenvectors, dim=-1, 
                                      index=idx.unsqueeze(-2).expand_as(eigenvectors))
            
            
            principal = evecs_sorted[..., 2]  # (B,3)
            
            
            norm = principal.norm(dim=-1, keepdim=True)
            principal = torch.where(norm > 1e-8, principal / norm, torch.zeros_like(principal))
            
            
            principal = torch.where(low_gradient_mask.unsqueeze(-1), 
                                  torch.zeros_like(principal), principal)
            evals_sorted = torch.where(low_gradient_mask.unsqueeze(-1), 
                                     torch.zeros_like(evals_sorted), evals_sorted)
            
        except Exception as e:
            if self._warned < 3:
                print(f"Warning: eigendecomposition failed; falling back to SVD: {e}")
                self._warned += 1
            
            try:
                with torch.cuda.amp.autocast(enabled=False):
                    U, S, Vh = torch.linalg.svd(T_reg)   # (B,3,3), (B,3), (B,3,3)
                principal = Vh[..., -1]              
                evals_sorted = S                     
                
                
                principal = torch.where(low_gradient_mask.unsqueeze(-1), 
                                      torch.zeros_like(principal), principal)
                evals_sorted = torch.where(low_gradient_mask.unsqueeze(-1), 
                                         torch.zeros_like(evals_sorted), evals_sorted)
            except Exception as svd_err:
                if self._warned < 6:
                    print(f"Warning: SVD also failed; returning zero directions: {svd_err}")
                    self._warned += 1
                principal = torch.zeros((B, 3), dtype=T_reg.dtype, device=T_reg.device)
                evals_sorted = torch.zeros((B, 3), dtype=T_reg.dtype, device=T_reg.device)

        
        if len(output_shape) == 5:  # (N, C, D, H, W)
            principal = principal.to(original_dtype).view(*output_shape, 3)
            evals_sorted = evals_sorted.to(original_dtype).view(*output_shape, 3)
        else:  # (N, D, H, W)
            principal = principal.to(original_dtype).view(*output_shape, 3)
            evals_sorted = evals_sorted.to(original_dtype).view(*output_shape, 3)
        
        
        principal = torch.where(torch.isfinite(principal), principal, torch.zeros_like(principal))
        evals_sorted = torch.where(torch.isfinite(evals_sorted), evals_sorted, torch.zeros_like(evals_sorted))
        
        return principal, evals_sorted
    
    def compute_coherence(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Implementation detail for the GR-GCDCN model."""
        
        lambda1 = eigenvalues[..., 0]  
        lambda2 = eigenvalues[..., 1]  
        lambda3 = eigenvalues[..., 2]  
        
        
        numerator = lambda2 - lambda3
        denominator = lambda1 + lambda2 + lambda3 + self.eps
        
        coherence = numerator / denominator
        
        return coherence
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation detail for the GR-GCDCN model."""
        profiling = self.__class__._profiling_enabled
        start_time = None
        if profiling:
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            start_time = time.perf_counter()

        
        
        x1 = x.mean(dim=1, keepdim=True)  # (N,1,D,H,W)
        
        
        gx, gy, gz = self.compute_gradients(x1)  
        
        
        T_xx = (gx * gx).squeeze(1)  
        T_xy = (gx * gy).squeeze(1)
        T_xz = (gx * gz).squeeze(1)
        T_yy = (gy * gy).squeeze(1)
        T_yz = (gy * gz).squeeze(1)
        T_zz = (gz * gz).squeeze(1)
        
        N, D, H, W = T_xx.shape
        
        
        if self.window_size > 1 and min(D, H, W) >= self.window_size:
            pad = self.window_size // 2
            
            avg3d = lambda t: F.avg_pool3d(t.unsqueeze(1), self.window_size, 1, pad).squeeze(1)
            T_xx, T_xy, T_xz = avg3d(T_xx), avg3d(T_xy), avg3d(T_xz)
            T_yy, T_yz, T_zz = avg3d(T_yy), avg3d(T_yz), avg3d(T_zz)
        
        
        T = torch.zeros(N, D, H, W, 3, 3, device=x.device, dtype=x.dtype)
        T[..., 0, 0] = T_xx  # T_xx
        T[..., 0, 1] = T_xy  # T_xy
        T[..., 0, 2] = T_xz  # T_xz
        T[..., 1, 0] = T_xy  # T_yx = T_xy
        T[..., 1, 1] = T_yy  # T_yy
        T[..., 1, 2] = T_yz  # T_yz
        T[..., 2, 0] = T_xz  # T_zx = T_xz
        T[..., 2, 1] = T_yz  # T_zy = T_yz
        T[..., 2, 2] = T_zz  # T_zz
        
        
        with torch.no_grad():
            principal_dirs, eigenvalues = self.extract_principal_direction_with_eigenvalues(T)
        coherence = self.compute_coherence(eigenvalues)
        
        
        # principal_dirs: (N, D, H, W, 3) -> (N, 1, D, H, W, 3)
        # coherence: (N, D, H, W) -> (N, 1, D, H, W)
        principal_dirs = principal_dirs.unsqueeze(1)
        coherence = coherence.unsqueeze(1)
        
        
        C = x.shape[1]
        principal_dirs = principal_dirs.expand(-1, C, -1, -1, -1, -1)  # (N, C, D, H, W, 3)
        coherence = coherence.expand(-1, C, -1, -1, -1)  # (N, C, D, H, W)
        
        if profiling and start_time is not None:
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            records = self.__class__._profiling_records
            if (
                self.__class__._profiling_max_samples is None
                or len(records) < self.__class__._profiling_max_samples
            ):
                records.append(elapsed_ms)
            if (
                self.__class__._profiling_max_samples is not None
                and len(records) >= self.__class__._profiling_max_samples
            ):
                self.__class__._profiling_enabled = False

        return principal_dirs, coherence


class GeometricConstraintLoss(nn.Module):
    """Implementation detail for the GR-GCDCN model."""
    
    def __init__(self, eps: float = 1e-6, 
                 use_mask_weighting: bool = True, 
                 use_coherence_weighting: bool = True,
                 mask_weight: float = 1.0,
                 coherence_weight: float = 1.0):
        super().__init__()
        self.eps = eps
        self.use_mask_weighting = use_mask_weighting
        self.use_coherence_weighting = use_coherence_weighting
        self.mask_weight = mask_weight
        self.coherence_weight = coherence_weight
    
    def forward(self, offsets: torch.Tensor, principal_dirs: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, coherence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Implementation detail for the GR-GCDCN model."""
        
        v = F.normalize(principal_dirs[:, :1], dim=-1)  # (N, 1, D, H, W, 3)
        
        
        # v: (N, 1, D, H, W, 3) -> (N, D, H, W, 1, 1, 3)
        v = v.squeeze(1)  # (N, D, H, W, 3)
        v = v.unsqueeze(4).unsqueeze(5)  # (N, D, H, W, 1, 1, 3)
        
        
        u = offsets / (torch.norm(offsets, dim=-1, keepdim=True) + self.eps)
        
        
        cos = (u * v).sum(dim=-1)  # (N, D, H, W, g, P)
        
        
        ell = 1.0 - torch.abs(cos)  # (N, D, H, W, g, P)
        
        
        if self.use_mask_weighting and mask is not None:
            
            w = mask / (mask.sum(dim=-1, keepdim=True) + self.eps)  # (N, D, H, W, g, P)
            ell = ell * (w * self.mask_weight)  
        
        
        ell = ell.mean(dim=-1)  # (N, D, H, W, g)
        
        
        if self.use_coherence_weighting and coherence is not None:
            
            coh = coherence[:, :1, :, :, :]  # (N, 1, D, H, W)
            coh = coh.squeeze(1)  # (N, D, H, W)
            coh = coh.unsqueeze(-1)  # (N, D, H, W, 1)
            ell = ell * (coh * self.coherence_weight)  
        
        
        return torch.mean(ell)


class DCNv3_3D_Enhanced(DCNv3_3D):
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
        return_offsets: bool = False,
    ) -> None:
        super().__init__(
            channels=channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad=pad,
            dilation=dilation,
            group=group,
            act_layer=act_layer,
            norm_layer=norm_layer,
            offset_scale_warmup_start=offset_scale_warmup_start,
            warmup_steps=warmup_steps,
            offset_axis_scale=offset_axis_scale,
            enable_monitor=enable_monitor,
        )
        self.return_offsets = return_offsets
        self._last_offsets: Optional[torch.Tensor] = None
        
        
        self.register_buffer(
            'offset_axis_scale_tensor',
            torch.tensor(offset_axis_scale, dtype=torch.float32)
        )
    
    def forward(self, x: torch.Tensor, return_offsets: Optional[bool] = None) -> torch.Tensor:
        """Implementation detail for the GR-GCDCN model."""
        
        should_return_offsets = return_offsets if return_offsets is not None else self.return_offsets
        
        
        N, C, D, H, W = x.shape
        assert C == self.channels, f"Input channels {C} do not match model channels {self.channels}."

        
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

        
        if should_return_offsets:
            
            D_p = D + 2 * self.pad
            H_p = H + 2 * self.pad
            W_p = W + 2 * self.pad
            
            self._last_offsets = self._get_normalized_offsets(offset, N, D_out, H_out, W_out, g, P, D_p, H_p, W_p)
            self._last_mask = mask.view(N, D_out, H_out, W_out, g, P)

        x_out = self._dcn_v3_core(x_proj, offset, mask)
        x_out = self.output_proj(x_out).permute(0, 4, 1, 2, 3)  # (N,C_out,D,H,W)
        
        if should_return_offsets:
            return x_out, self._last_offsets
        else:
            return x_out
    
    def _get_normalized_offsets(self, offset: torch.Tensor, N: int, D_out: int, H_out: int, W_out: int, 
                               g: int, P: int, D_p: int, H_p: int, W_p: int) -> torch.Tensor:
        """Implementation detail for the GR-GCDCN model."""
        
        offset_reshaped = offset.view(N, D_out, H_out, W_out, g, P, 3)
        
        
        norm = torch.tensor([W_p, H_p, D_p], device=offset.device, dtype=offset.dtype).view(1, 1, 1, 1, 1, 1, 3)
        offset_normalized = offset_reshaped / norm
        
        
        
        axis = self.offset_axis_scale_tensor.to(dtype=offset.dtype).view(1, 1, 1, 1, 1, 1, 3)
        cur_scale = self.get_current_offset_scale()
        
        
        normalized_offsets = offset_normalized * axis * float(cur_scale)
        
        return normalized_offsets
    
    def get_last_offsets(self) -> Optional[torch.Tensor]:
        """Implementation detail for the GR-GCDCN model."""
        return self._last_offsets
    
    def get_last_mask(self) -> Optional[torch.Tensor]:
        """Implementation detail for the GR-GCDCN model."""
        return getattr(self, '_last_mask', None)
    
    def clear_offsets(self):
        """Implementation detail for the GR-GCDCN model."""
        self._last_offsets = None


class DCNRefine3D_Enhanced(nn.Module):
    """Implementation detail for the GR-GCDCN model."""
    
    def __init__(self, channels, groups=2, k=3, axis_scale=(0.5,1.0,1.0), return_offsets=False, use_residual_gate=True):
        super().__init__()
        self.use_residual_gate = use_residual_gate
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.pre = nn.Conv3d(channels, channels, 1, bias=False)
        self.return_offsets = return_offsets
        
        
        def choose_group(channels, default_group=groups):
            if channels % default_group == 0:
                return default_group
            elif channels % 4 == 0:
                return 4
            elif channels % 2 == 0:
                return 2
            else:
                return 1
        
        selected_group = choose_group(channels, groups)
            
        self.dcn = DCNv3_3D_Enhanced(
            channels=channels, 
            out_channels=channels,
            kernel_size=k, 
            pad=1, 
            group=selected_group,
            norm_layer='IN', 
            act_layer='GELU',
            offset_scale_warmup_start=0.0, 
            warmup_steps=2000,
            offset_axis_scale=axis_scale, 
            enable_monitor=True,
            return_offsets=return_offsets
        )
        self.post = nn.Conv3d(channels, channels, 1, bias=False)
        self._last_offsets = None

    def forward(self, x, return_offsets=None):
        """Implementation detail for the GR-GCDCN model."""
        should_return_offsets = return_offsets if return_offsets is not None else self.return_offsets
        
        pre_out = self.pre(x)
        
        if should_return_offsets:
            dcn_out, offsets = self.dcn(pre_out, return_offsets=True)
            self._last_offsets = offsets
            self._last_mask = self.dcn.get_last_mask()
        else:
            dcn_out = self.dcn(pre_out)
        
        y = self.post(dcn_out)
        
        if self.use_residual_gate:
            output = x + torch.sigmoid(self.gate) * y
        else:
            output = y

        if should_return_offsets:
            return output, self._last_offsets
        else:
            return output
    
    def get_last_offsets(self):
        """Implementation detail for the GR-GCDCN model."""
        return self._last_offsets
    
    def get_last_mask(self):
        """Implementation detail for the GR-GCDCN model."""
        return getattr(self, '_last_mask', None)
    
    def step(self, n=1):
        """Implementation detail for the GR-GCDCN model."""
        self.dcn.step(n)

    def set_residual_gate(self, enabled: bool):
        self.use_residual_gate = bool(enabled)


def create_enhanced_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4):
    """Implementation detail for the GR-GCDCN model."""
    param_groups = _offset_mask_param_groups(model, lr, weight_decay, offset_lr_scale=0.25)
    return torch.optim.AdamW(param_groups)


def clip_dcn_offset_mask(module: nn.Module, max_norm: float = 1.0) -> None:
    """Implementation detail for the GR-GCDCN model."""
    from .DCNv3_3D import clip_offset_mask_grads
    clip_offset_mask_grads(module, max_norm)
