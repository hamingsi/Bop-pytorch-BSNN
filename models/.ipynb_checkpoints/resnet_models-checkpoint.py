import random
from models.layers import *
from models.quant_function import *


# ------------------- #
#   ResNet Example    #
# ------------------- #

fr = []

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return IRConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, step=2):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.lif1 = LIFSpike()

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.lif2 = LIFSpike()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        global fr

        identity = x

        out = self.conv1_s(x)
        out = self.lif1(out)
        fr.append(out.mean())

        out = self.conv2_s(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.lif2(out.clone())
        fr.append(out.mean())

        return out

class ResNet_CIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10, input_c=3, norm_layer=None, step=2):
        super(ResNet_CIFAR, self).__init__()
        self.T = step

        if norm_layer is None:
            norm_layer = tdBatchNorm
        self._norm_layer = norm_layer

        inplanes = 128
        self.inplanes = 128
        # self.conv1 = nn.Conv2d(input_c, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = first_conv(input_c, self.inplanes, 3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.lif1 = LIFSpike()

        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=2)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        # self.fc = ClassifyLinear(nn.Linear(inplanes * 4 * block.expansion, num_classes))
        self.fc = ClassifyLinear(last_fc(inplanes * 4 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()
            elif isinstance(m, IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                IRConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, self.T))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, step=self.T))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        global fr
        fr = []

        x = self.conv1_s(x)
        x = self.lif1(x)
        fr.append(x.mean())

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x, fr

    def forward(self, x):
        x = add_dimention(x, self.T)
        return self._forward_impl(x)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, input_c=3, norm_layer=None, step=2):
        super(ResNet, self).__init__()
        self.T = step

        if norm_layer is None:
            norm_layer = tdBatchNorm
        self._norm_layer = norm_layer

        inplanes = 64
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(input_c, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = first_conv(input_c, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.lif1 = LIFSpike()
        self.maxpool = SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes*8, layers[3], stride=2)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        # self.fc = ClassifyLinear(nn.Linear(inplanes * 4 * block.expansion, num_classes))
        self.fc = ClassifyLinear(last_fc(inplanes * 8 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()
            elif isinstance(m, IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                IRConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, self.T))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, step=self.T))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        global fr
        fr = []

        x = self.conv1_s(x)
        x = self.lif1(x)
        fr.append(x.mean())

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x, fr

    def forward(self, x):
        x = add_dimention(x, self.T)
        return self._forward_impl(x)



def resnet19(**kwargs):
    # for CIFAR
    model = ResNet_CIFAR(BasicBlock, [3, 3, 2], **kwargs)
    return model


def resnet18(**kwargs):
    # for ImageNet
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

if __name__ == '__main__':
    model = resnet19(num_classes=10)
    model.T = 3
    x = torch.rand(2,3,32,32)
    y = model(x)
    print(y.shape)