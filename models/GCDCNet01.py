import torch
from torch import nn
from models.blocks.GCDCNv3 import DCNv3_3D_Enhanced, StructureTensor3D, GeometricConstraintLoss, DCNRefine3D_Enhanced


class DoubleConv(nn.Module):
    """Double 3x3x3 convolution block with IN and ReLU, as per 3D U-Net paper."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvWithDCNv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        def choose_group(channels):
            if channels % 4 == 0:
                return 4
            elif channels % 2 == 0:
                return 2
            else:
                return 1
        
        group1 = choose_group(in_channels)
        group2 = choose_group(out_channels)
        
        
        self.double_conv = nn.Sequential(
            DCNv3_3D_Enhanced(channels=in_channels, out_channels=out_channels,
                             kernel_size=3, pad=1, group=group1, norm_layer='IN', act_layer='GELU',
                             enable_monitor=True),
            DCNv3_3D_Enhanced(channels=out_channels, out_channels=out_channels,
                             kernel_size=3, pad=1, group=group2, norm_layer='IN', act_layer='GELU',
                             enable_monitor=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Encoder block: DoubleConv + MaxPooling, doubles channels before pooling."""
    def __init__(self, in_channels, out_channels, use_dcn=False,
                 use_mask_weighting=True, use_coherence_weighting=True,
                 mask_weight=1.0, coherence_weight=1.0,
                 structure_use_separable=True):
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        
        if use_dcn:
            
            self.refine = DCNRefine3D_Enhanced(out_channels)
            
            self.structure_tensor = StructureTensor3D(use_separable=structure_use_separable)
            self.geometric_loss_fn = GeometricConstraintLoss(
                use_mask_weighting=use_mask_weighting,
                use_coherence_weighting=use_coherence_weighting,
                mask_weight=mask_weight,
                coherence_weight=coherence_weight
            )
            self._last_geometric_loss = None
        else:
            self.refine = nn.Identity()
            
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x, return_geometric_loss=False):
        conv_out = self.conv(x)  # Skip connection output
        
        if isinstance(self.refine, DCNRefine3D_Enhanced):
            
            if return_geometric_loss:
                
                
                with torch.no_grad():
                    principal_dirs, coherence = self.structure_tensor(conv_out)
                
                conv_out, offsets = self.refine(conv_out, return_offsets=True)
                mask = self.refine.get_last_mask()
                
                
                if offsets is not None:
                    geometric_loss = self.geometric_loss_fn(
                        offsets, 
                        principal_dirs, 
                        mask=mask, 
                        coherence=coherence
                    )
                    self._last_geometric_loss = geometric_loss
                else:
                    self._last_geometric_loss = torch.tensor(0.0, device=x.device)
            else:
                
                conv_out = self.refine(conv_out)
        else:
            conv_out = self.refine(conv_out)  # DCN refinement
        
        pool_out = self.pool(conv_out)
        
        if return_geometric_loss and isinstance(self.refine, DCNRefine3D_Enhanced):
            return conv_out, pool_out, self._last_geometric_loss
        else:
            return conv_out, pool_out
    
    def get_last_geometric_loss(self):
        """Implementation detail for the GR-GCDCN model."""
        return getattr(self, '_last_geometric_loss', None)


class EncoderBlockWithDCNv3(nn.Module):
    """Encoder block: DoubleConv + MaxPooling, doubles channels before pooling."""
    def __init__(self, in_channels, out_channels, use_dcn_refine=True):
        super(EncoderBlockWithDCNv3, self).__init__()
        self.conv = DoubleConvWithDCNv3(in_channels, out_channels)
        self.refine = DCNRefine3D(out_channels) if use_dcn_refine else nn.Identity()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)  # Skip connection output
        conv_out = self.refine(conv_out)  # DCN refinement
        pool_out = self.pool(conv_out)
        return conv_out, pool_out


class DCNRefine3D(nn.Module):
    def __init__(self, channels, groups=2, k=3, axis_scale=(0.5,1.0,1.0)):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))          
        self.pre  = nn.Conv3d(channels, channels, 1, bias=False)
        
        
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
        
        self.dcn  = DCNv3_3D_Enhanced(channels=channels, out_channels=channels,
                                      kernel_size=k, pad=1, group=selected_group,
                                      norm_layer='IN', act_layer='GELU',
                                      offset_scale_warmup_start=0.0, warmup_steps=2000,
                                      offset_axis_scale=axis_scale, enable_monitor=True)
        self.post = nn.Conv3d(channels, channels, 1, bias=False)

    def forward(self, x):
        y = self.post(self.dcn(self.pre(x)))
        return x + torch.sigmoid(self.gate) * y


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_dcn=False,
                 use_mask_weighting=True, use_coherence_weighting=True,
                 mask_weight=1.0, coherence_weight=1.0,
                 structure_use_separable=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, 2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)
        
        if use_dcn:
            
            self.refine = DCNRefine3D_Enhanced(out_ch)
            
            self.structure_tensor = StructureTensor3D(use_separable=structure_use_separable)
            self.geometric_loss_fn = GeometricConstraintLoss(
                use_mask_weighting=use_mask_weighting,
                use_coherence_weighting=use_coherence_weighting,
                mask_weight=mask_weight,
                coherence_weight=coherence_weight
            )
            self._last_geometric_loss = None
        else:
            self.refine = nn.Identity()

    def forward(self, x, skip, return_geometric_loss=False):
        x = self.up(x)
        
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                 diffY // 2, diffY - diffY // 2,
                                 diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        if isinstance(self.refine, DCNRefine3D_Enhanced):
            
            if return_geometric_loss:
                
                
                with torch.no_grad():
                    principal_dirs, coherence = self.structure_tensor(x)
                
                x, offsets = self.refine(x, return_offsets=True)
                mask = self.refine.get_last_mask()
                
                
                if offsets is not None:
                    geometric_loss = self.geometric_loss_fn(
                        offsets, 
                        principal_dirs, 
                        mask=mask, 
                        coherence=coherence
                    )
                    self._last_geometric_loss = geometric_loss
                else:
                    self._last_geometric_loss = torch.tensor(0.0, device=x.device)
                
                return x, self._last_geometric_loss
            else:
                
                x = self.refine(x)
        else:
            x = self.refine(x)
        
        return x
    
    def get_last_geometric_loss(self):
        """Implementation detail for the GR-GCDCN model."""
        return getattr(self, '_last_geometric_loss', None)



# class DecoderBlock(nn.Module):
#     """Decoder block: UpConv + Concat + DoubleConv, halves channels after concat."""
#     def __init__(self, in_channels, skip_channels, out_channels):
#         super(DecoderBlock, self).__init__()
#         self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv = DoubleConv(out_channels + skip_channels, out_channels)

#     def forward(self, x, skip):
#         x = self.upconv(x)
#         # Pad to match skip connection size
#         diffZ = skip.size()[2] - x.size()[2]
#         diffY = skip.size()[3] - x.size()[3]
#         diffX = skip.size()[4] - x.size()[4]
#         x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
#                                  diffY // 2, diffY - diffY // 2,
#                                  diffZ // 2, diffZ - diffZ // 2])
#         x = torch.cat([x, skip], dim=1)
#         return self.conv(x)

class DecoderBlockWithDCNv3(nn.Module):
    """Decoder block: UpConv + Concat + DoubleConv, halves channels after concat."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlockWithDCNv3, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvWithDCNv3(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        # Pad to match skip connection size
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                 diffY // 2, diffY - diffY // 2,
                                 diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Bottleneck(nn.Module):
    """Bottleneck: DoubleConv at the bottom, doubles channels."""
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)

class BottleneckWithDCNv3(nn.Module):
    """Bottleneck: DoubleConvWithDCNv3 at the bottom, doubles channels."""
    def __init__(self, in_channels, out_channels):
        super(BottleneckWithDCNv3, self).__init__()
        self.conv = DoubleConvWithDCNv3(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)



class OutputConv(nn.Module):
    """Final 1x1x1 convolution for segmentation output (always returns logits)."""
    def __init__(self, in_channels, out_channels):
        super(OutputConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        
        return self.conv(x)


class GCDCNet(nn.Module):
    """3D U-Net with modular design, aligned with FaultSeg3D channels."""
    def __init__(self, num_channels=1, num_classes=2, geometric_loss_weight=0.1,
                 use_mask_weighting=True, use_coherence_weighting=True,
                 mask_weight=1.0, coherence_weight=1.0,
                 use_separable_sobel=True):
        super(GCDCNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.geometric_loss_weight = geometric_loss_weight
        
        
        self.use_mask_weighting = use_mask_weighting
        self.use_coherence_weighting = use_coherence_weighting
        self.mask_weight = mask_weight
        self.coherence_weight = coherence_weight

        # Channel counts aligned with FaultSeg3D
        enc_channels = [16, 32, 64, 128]  # Encoder output channels
        dec_channels = [128, 64, 32, 16]  # Decoder output channels

        
        self.enc1 = EncoderBlock(num_channels, enc_channels[0], structure_use_separable=use_separable_sobel)
        self.enc2 = EncoderBlock(enc_channels[0], enc_channels[1], use_dcn=True,
                                use_mask_weighting=use_mask_weighting, use_coherence_weighting=use_coherence_weighting,
                                mask_weight=mask_weight, coherence_weight=coherence_weight,
                                structure_use_separable=use_separable_sobel)
        self.enc3 = EncoderBlock(enc_channels[1], enc_channels[2], use_dcn=True,
                                use_mask_weighting=use_mask_weighting, use_coherence_weighting=use_coherence_weighting,
                                mask_weight=mask_weight, coherence_weight=coherence_weight,
                                structure_use_separable=use_separable_sobel) 
        self.enc4 = EncoderBlock(enc_channels[2], enc_channels[3], use_dcn=True,
                                use_mask_weighting=use_mask_weighting, use_coherence_weighting=use_coherence_weighting,
                                mask_weight=mask_weight, coherence_weight=coherence_weight,
                                structure_use_separable=use_separable_sobel) 

        # Bottleneck
        self.bottleneck = Bottleneck(enc_channels[3], enc_channels[3] * 2)

        
        self.dec4 = DecoderBlock(enc_channels[3] * 2, enc_channels[3], dec_channels[0], use_dcn=True,
                                use_mask_weighting=use_mask_weighting, use_coherence_weighting=use_coherence_weighting,
                                mask_weight=mask_weight, coherence_weight=coherence_weight,
                                structure_use_separable=use_separable_sobel)
        self.dec3 = DecoderBlock(dec_channels[0], enc_channels[2], dec_channels[1], use_dcn=True,
                                use_mask_weighting=use_mask_weighting, use_coherence_weighting=use_coherence_weighting,
                                mask_weight=mask_weight, coherence_weight=coherence_weight,
                                structure_use_separable=use_separable_sobel)
        self.dec2 = DecoderBlock(dec_channels[1], enc_channels[1], dec_channels[2], use_dcn=True,
                                use_mask_weighting=use_mask_weighting, use_coherence_weighting=use_coherence_weighting,
                                mask_weight=mask_weight, coherence_weight=coherence_weight,
                                structure_use_separable=use_separable_sobel)
        self.dec1 = DecoderBlock(dec_channels[2], enc_channels[0], dec_channels[3])

        # Output
        self.output = OutputConv(dec_channels[3], num_classes)
        
        
        self._geometric_losses = []

    def forward(self, x, return_geometric_loss=False):
        """Implementation detail for the GR-GCDCN model."""
        self._geometric_losses = []
        
        # Encoder
        skip1, x = self.enc1(x)  # 128x128x128, 16; 64x64x64, 16
        
        if return_geometric_loss:
            skip2, x, geo_loss2 = self.enc2(x, True)  # return_geometric_loss=True
            skip3, x, geo_loss3 = self.enc3(x, True)  # return_geometric_loss=True
            skip4, x, geo_loss4 = self.enc4(x, True)  # return_geometric_loss=True
            self._geometric_losses.extend([geo_loss2, geo_loss3, geo_loss4])
        else:
            skip2, x = self.enc2(x)  # 64x64x64, 32; 32x32x32, 32
            skip3, x = self.enc3(x)  # 32x32x32, 64; 16x16x16, 64
            skip4, x = self.enc4(x)  # 16x16x16, 128; 8x8x8, 128

        # Bottleneck
        x = self.bottleneck(x)  # 8x8x8, 256

        # Decoder
        if return_geometric_loss:
            x, geo_loss_dec4 = self.dec4(x, skip4, True)  # return_geometric_loss=True
            x, geo_loss_dec3 = self.dec3(x, skip3, True)  # return_geometric_loss=True
            x, geo_loss_dec2 = self.dec2(x, skip2, True)  # return_geometric_loss=True
            x = self.dec1(x, skip1)  # 128x128x128, 16
            self._geometric_losses.extend([geo_loss_dec4, geo_loss_dec3, geo_loss_dec2])
        else:
            x = self.dec4(x, skip4)  # 16x16x16, 128
            x = self.dec3(x, skip3)  # 32x32x32, 64
            x = self.dec2(x, skip2)  # 64x64x64, 32
            x = self.dec1(x, skip1)  # 128x128x128, 16

        # Output
        x = self.output(x)  # 128x128x128, 2
        
        if return_geometric_loss:
            
            if self._geometric_losses:
                
                scalar_losses = []
                for loss in self._geometric_losses:
                    if loss.dim() == 0:
                        scalar_losses.append(loss)
                    else:
                        scalar_losses.append(loss.mean())  
                
                
                total_geometric_loss = torch.stack(scalar_losses).mean()
            else:
                total_geometric_loss = torch.tensor(0.0, device=x.device)
            
            
            if total_geometric_loss.dim() == 0:
                total_geometric_loss = total_geometric_loss.unsqueeze(0)  # shape: (1,)
            
            return x, total_geometric_loss
        else:
            return x
    
    def get_geometric_loss(self, x):
        """Implementation detail for the GR-GCDCN model."""
        _, geometric_loss = self.forward(x, True)  # return_geometric_loss=True
        return geometric_loss  
    
    def step_warmup(self, n=1):
        """Implementation detail for the GR-GCDCN model."""
        for module in self.modules():
            if hasattr(module, 'step'):
                module.step(n)
