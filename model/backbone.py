import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def ResNet34(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    512, 64
    """
    print(in_c)
    model = ResNet(BasicBlock, [3, 4, 6, 3], output_stride, BatchNorm, in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet34'])
    return model


def ResNet18(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    512, 256, 128, 64, 64
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm, in_c=in_c)
    if in_c !=3:
        pretrained=False
    # 是否加载预训练模型
    if pretrained:
        print("加载resnet18预训练模型...")
        model._load_pretrained_model(model_urls['resnet18'])
    return model


def ResNet50(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    2048, 256
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, in_c=in_c)
    if in_c !=3:
        pretrained=False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet50'])
    return model


'''
    ResNet 中的基础残差块结构    
'''
class BasicBlock(nn.Module):
    expansion = 1 # 通道扩展系数，表示该块输出通道数相对于输入通道数的倍数（BasicBlock 中不扩展通道）

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()

        # 3x3 卷积，可能包含下采样（stride>1）或空洞卷积（dilation>1）
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        # 第一个归一化层
        self.bn1 = BatchNorm(planes)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        # self.do1 = nn.Dropout2d(p=0.2)
        # 进一步提取特征，保持空间尺寸和通道数不变
        self.conv2 = conv3x3(planes, planes)
        # 第二个归一化层
        self.bn2 = BatchNorm(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x # 保存输入 x 的原始值，用于残差连接。

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

'''
    改变输入和输出的通道数的
'''
class PA(nn.Module):
    def __init__(self, inchan = 512, out_chan = 32):
        super().__init__()
        self.conv = nn.Conv2d(inchan, out_chan, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.re = nn.ReLU()
        self.do = nn.Dropout2d(0.2)

        self.pa_conv = nn.Conv2d(out_chan, out_chan, kernel_size=1, padding=0, groups=out_chan)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.conv(x)
        x = self.do(self.re(self.bn(x0)))
        return x0 *self.sigmoid(self.pa_conv(x))


class ResNet(nn.Module):

    def __init__(self,  block, layers, output_stride, BatchNorm, pretrained=True, in_c=3):

        self.inplanes = 64
        self.in_c = in_c
        print('in_c: ',self.in_c)
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        elif output_stride == 16: # 这篇论文走的是这个
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        elif output_stride == 4:
            strides = [1, 1, 1, 1]
            dilations = [1, 2, 4, 8]
        else:
            raise NotImplementedError

        # Modules
        #
        # Conv 7 * 7的卷积：输入 B 3 H W  输出 B 64 H/2 W/2
        self.conv1 = nn.Conv2d(self.in_c, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64) # 归一化，输入和输出格式一致
        self.relu = nn.ReLU(inplace=True) # 激活，输入和输出格式一致
        # 最大池化技术，输入 B 64 H/2 W/2,输出 B 64 h/4 w/4
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4个ResBlock块
        # resbolck1：输入是 B 64 H/4 W/4  输出是 B 64 H/4 W/4
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        # resbolck2：输入是 B 64 H/4 W/4  输出是 B 128 H/8 W/8
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        # resbolck3：输入是 B 128 H/8 W/8  输出是 B 256 H/8 W/8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        # resblock4：输入是B 256 H/8 W/8 输出是 B 512 H/16 W/16
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        # 初始化权重
        self._init_weight()

        #
        self.pos_s16 = PA(512, 32)
        self.pos_s8 = PA(128, 32)
        self.pos_s4 = PA(64, 32)

    '''
        params:
            block=BasicBlock,       # 使用的残差块类型
            planes=64,             # 目标基础通道数,输出通道数实际上是planes * block.expansion
            blocks=layers[0],      # 该层的block数量（如ResNet18中layers[0]=2）
            stride=strides[0],     # 第一个block的步长（如1）
            dilation=dilations[0],  # 空洞率（如1）
            BatchNorm=BatchNorm     # 归一化层类型
    '''
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                #
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)  # |2 conv7 * 7的卷积核 得到 b 64 h/2 w/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # |4 输入 b 64 h/2 w/4  输出 b 64 h/4 w/4


        x = self.layer1(x)  # | 4 输入 b 64 h/4 w/4  输出 b 64 h/4 w/4
        low_level_feat2 = x

        x = self.layer2(x)  # | 8 输入 b 64 h/4 w/4  输出 b 128 h/8 w/8
        low_level_feat3 = x

        x = self.layer3(x)  # | 8 输入 b 128 h/8 w/8  输出 b 256 h/8 w/8
        x = self.layer4(x)  # | 16 输入 b 256 h/8 w/8  输出 b 512 h/16 w/16

        out_s16, out_s8, out_s4 = self.pos_s16(x), self.pos_s8(low_level_feat3), self.pos_s4(low_level_feat2)
        return out_s16, out_s8, out_s4


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def build_backbone(backbone, output_stride, BatchNorm, in_c=3):
    if backbone == 'resnet50':
        return ResNet50(output_stride, BatchNorm, in_c=in_c)
    elif backbone == 'resnet34':
        return ResNet34(output_stride, BatchNorm, in_c=in_c)
    elif backbone == 'resnet18':
        return ResNet18(output_stride, BatchNorm, in_c=in_c)
    else:
        raise NotImplementedError

