import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .modules import TransformerDecoder, Transformer
from einops import rearrange


'''
    是一个 基于 Transformer 的视觉 Token 编码器，用于将空间特征图转换为全局上下文 Token
'''
class token_encoder(nn.Module):
    '''
        params
            in_chan	输入特征通道数（默认 32）。
            token_len	Token 数量（默认 4），控制上下文信息的压缩程度。
            heads	Transformer 多头注意力头数（默认 8）。
    '''
    def __init__(self, in_chan = 32, token_len = 4, heads = 8):
        super(token_encoder, self).__init__()
        self.token_len = token_len
        # 通过 1×1 卷积生成 token_len 个空间注意力图。
        # 输入：B C H W  输出 B token_len H W
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        # 位置编码：为 Token 添加可学习的位置信息（类似 ViT 的 [CLS] Token）
        # pos_embedding 形状：[1, token_len, C]，广播机制自动将[]
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len, in_chan))
        # transformer编码器，dim就是输入的通道数
        self.transformer = Transformer(dim=in_chan, depth=1, heads=heads, dim_head=64, mlp_dim=64, dropout=0)

    def forward(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x) ## [B, token_len, H, W]
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous() # [B, token_len, H*W]
        spatial_attention = torch.softmax(spatial_attention, dim=-1) # 归一化，输入和输出一致
        # 特征图展平
        x = x.view([b, c, -1]).contiguous() # B C H*W

        # einsum运算
        # 输入 spatial_attention[B, token_len, H*W]  x[ B C H*W]  输出：[B, token_len, C]
        tokens = torch.einsum('bln, bcn->blc', spatial_attention, x)
        # pos_embedding 形状：[1, token_len, C]，广播机制自动将[B,token_len,C]
        tokens += self.pos_embedding
        x = self.transformer(tokens)
        return x

class token_decoder(nn.Module):
    def __init__(self, in_chan = 32, size = 32, heads = 8):
        super(token_decoder, self).__init__()
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, in_chan, size, size))
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        # x指的是feature
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x


class context_aggregator(nn.Module):
    def __init__(self, in_chan=32, size=32):
        super(context_aggregator, self).__init__()
        self.token_encoder = token_encoder(in_chan=in_chan, token_len=4)
        self.token_decoder = token_decoder(in_chan = 32, size = size, heads = 8)

    def forward(self, feature):
        token = self.token_encoder(feature)
        out = self.token_decoder(feature, token)
        return out

class Classifier(nn.Module):
    def __init__(self, in_chan=32, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            # 除了通道数从 64->32 ，其他就是H W 不变
                            nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan), # 对输入数据进行批归一化，以提高训练的稳定性和速度。
                            nn.ReLU(), # ReLU 激活函数加入非线性，帮助网络学习复杂的特征。
                            # 除了通道数从 32->2 ，其他就是H W 不变
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1)) #
    def forward(self, x):
        x = self.head(x)
        return x

class CDNet(nn.Module):
    # backbone	基础特征提取网络（如 resnet18），输出多尺度特征。
    # output_stride	控制特征图下采样率（如 16 表示 H/16 × W/16）。
    # img_size	输入图像尺寸（如 512）。
    # chan_num	中间特征通道数（默认 32）。
    # n_class	分类类别数（2 表示二分类：变化/未变化）。
    def __init__(self,  backbone='resnet18', output_stride=16, img_size = 512, img_chan=3, chan_num = 32, n_class =2):
        super(CDNet, self).__init__()
        # 输入：(B, C, H, W)（批次大小, 通道数, 高度, 宽度）
        # 输出：相同形状 (B, C, H, W)，但对每个通道进行归一化。
        BatchNorm = nn.BatchNorm2d

        # 特征提取器的定义
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, img_chan)

        # msca 多尺度上下文聚合器的定义
        self.CA_s16 = context_aggregator(in_chan=chan_num, size=img_size//16)
        self.CA_s8 = context_aggregator(in_chan=chan_num, size=img_size//8)
        self.CA_s4 = context_aggregator(in_chan=chan_num, size=img_size//4)

        self.conv_s8 = nn.Conv2d(chan_num*2, chan_num, kernel_size=3, padding=1)
        self.conv_s4 = nn.Conv2d(chan_num*2, chan_num, kernel_size=3, padding=1)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)

        self.classifier1 = Classifier(n_class = n_class)
        self.classifier2 = Classifier(n_class = n_class)
        self.classifier3 = Classifier(n_class = n_class)


    def forward(self, img1, img2):
        # --------------------特征提取部分--------------------------
        # out1_s4[B,32,H/4,W/4] out1_s8[B,32,H/8,W/8] out1_s16[B,32,H/16,W/16]
        out1_s16, out1_s8, out1_s4 = self.backbone(img1) # 图片 1
        out2_s16, out2_s8, out2_s4 = self.backbone(img2) # 图片 2

        # --------------------context aggregate (scale 16, scale 8, scale 4)--------------------------
        # 输入out1_s16[B,32,H/16,W/16]    输出：x1_s16[B,32,H/16,W/16]
        x1_s16= self.CA_s16(out1_s16)
        x2_s16 = self.CA_s16(out2_s16)
        # 输入 x1_s16[B,32,H/16,W/16]  x2_s16[B,32,H/16,W/16]   输出：x16[B,64,H/16,W/16]
        x16 = torch.cat([x1_s16, x2_s16], dim=1)
        # 指定了输出图像的目标大小，它等于原始图像的高度和宽度 (H, W)。
        # 输入[B,64,H/16,W/16]  输出格式：(B, 64, H, W)
        x16 = F.interpolate(x16, size=img1.shape[2:], mode='bicubic', align_corners=True)
        # x16的格式变为 [B,2,H,W]
        x16 = self.classifier1(x16)

        # out1_s8格式为 [B,32,H/8,W/8]
        out1_s8 = self.conv_s8(torch.cat([self.upsamplex2(x1_s16), out1_s8], dim=1)) # 图片1
        out2_s8 = self.conv_s8(torch.cat([self.upsamplex2(x2_s16), out2_s8], dim=1)) # 图片2

        # x1_s8格式为[B,32,H/8,W/8]
        x1_s8 = self.CA_s8(out1_s8)
        x2_s8 = self.CA_s8(out2_s8)

        # x8的最终输出格式[B,2,H,W]
        x8 = torch.cat([x1_s8, x2_s8], dim=1)
        x8 = F.interpolate(x8, size=img1.shape[2:], mode='bicubic', align_corners=True)
        x8 = self.classifier2(x8)

        # out1_s4格式为[B,32,H/4,H/W]
        out1_s4 = self.conv_s4(torch.cat([self.upsamplex2(x1_s8), out1_s4], dim=1))
        out2_s4 = self.conv_s4(torch.cat([self.upsamplex2(x2_s8), out2_s4], dim=1))

        x1 = self.CA_s4(out1_s4)
        x2 = self.CA_s4(out2_s4)
        # x的最终输出格式:[B,2,H,W]
        x = torch.cat([x1, x2], dim=1)
        x = F.interpolate(x, size=img1.shape[2:], mode='bicubic', align_corners=True)
        x = self.classifier3(x)

        return x, x8, x16

    #
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
