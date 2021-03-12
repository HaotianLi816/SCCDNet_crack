from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
import math
input_size = (448, 448)

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class dilated_conv(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv_3x3 = nn.Conv2d(in_, in_, kernel_size=3, stride=1, padding=1, groups=in_)
        self.bn_conv3x3 = nn.BatchNorm2d(in_)
        self.conv_1x1 = nn.Conv2d(in_, out, kernel_size=1)
        self.bn_conv1x1 = nn.BatchNorm2d(out)
    def forward(self, x):
        out = F.relu(self.bn_conv1x1(self.conv_1x1(self.bn_conv3x3(self.conv_3x3(x)))))
        return out

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = dilated_conv(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Se_module_diff(nn.Module):
    def __init__(self, inp, oup, Avg_size = 1, se_ratio = 1):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((Avg_size, Avg_size))
        num_squeezed_channels = max(1,int(inp / se_ratio))
        self._se_reduce = nn.Conv2d(in_channels=inp, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        self.Avg_size = Avg_size
        self.reset_parameters()

    #x and z are different conv layer and z pass through more convs
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x, z):
        SIZE = z.size()
        y = self.avg(x)
        y = self._se_reduce(y)
        y = y * torch.sigmoid(y)
        y = self._se_expand(y)
        if self.Avg_size != 1:
            y = F.upsample_bilinear(y, size=[SIZE[2], SIZE[3]])
        z = torch.sigmoid(y) * z
        return z

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = 4
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, in_channels * mid_channels, kernel_size=3, stride=1, padding=3,
                                    dilation=3, groups=in_channels)
        self.bn_conv_3x3_1_1 = nn.BatchNorm2d(in_channels * mid_channels)
        # self.conv_3x3_1_point = nn.Conv2d(in_channels * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_3x3_1_point = nn.Conv2d(in_channels * mid_channels, out_channels, kernel_size=1)
        self.bn_conv_3x3_1_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, in_channels * mid_channels, kernel_size=3, stride=1, padding=7,
                                    dilation=7, groups=in_channels)
        self.bn_conv_3x3_2_1 = nn.BatchNorm2d(in_channels * mid_channels)
        # self.conv_3x3_2_point = nn.Conv2d(in_channels * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_3x3_2_point = nn.Conv2d(in_channels * mid_channels, out_channels, kernel_size=1)
        self.bn_conv_3x3_2_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, in_channels * mid_channels, kernel_size=3, stride=1, padding=11,
                                    dilation=11, groups=in_channels)
        self.bn_conv_3x3_3_1 = nn.BatchNorm2d(in_channels * mid_channels)
        # self.conv_3x3_3_point = nn.Conv2d(in_channels * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_3x3_3_point = nn.Conv2d(in_channels * mid_channels, out_channels, kernel_size=1)
        self.bn_conv_3x3_3_2 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
        self.conv_1x1_3 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)  # (160 = 5*32)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)
        self.conv_1x1_4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    def forward(self, x):
        feature_map_h = x.size()[2]  # (== h/16)
        feature_map_w = x.size()[3]  # (== w/16)
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))  # 111(shape: ([16, 256, 14, 14])
        out_3x3_1 = F.relu(self.bn_conv_3x3_1_2(
        self.conv_3x3_1_point(self.bn_conv_3x3_1_1(self.conv_3x3_1(x)))))  # (shape: ([16, 4, 14, 14])
        out_3x3_2 = F.relu(self.bn_conv_3x3_2_2(
        self.conv_3x3_2_point(self.bn_conv_3x3_2_1(self.conv_3x3_2(x)))))  # (shape: ([16, 4, 14, 14])
        out_3x3_3 = F.relu(self.bn_conv_3x3_3_2(
        self.conv_3x3_3_point(self.bn_conv_3x3_3_1(self.conv_3x3_3(x)))))  # (shape: [16, 4, 14, 14])

        # 把输出改为256
        # out_img = self.avg_pool(x)  # (shape: ([16, 256, 1, 1])
        # out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))  # (shape: [16, 4, 1, 1])
        # out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w),mode="bilinear")  # (shape: ([16, 4, 14, 14])

        x = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3],1)  # (shape: ([16, 20, 14, 14])
        x = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(x)))  # (shape: [16, 4, 14, 14])
        x_out = self.conv_1x1_4(x)  #[16, 256, 14, 14]

        return x_out



class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        #print(torchvision.models.vgg16(pretrained=pretrained))

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        # self.dilated_conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #                                    self.relu)
        # self.dilated_conv2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #                                    self.relu)
        # self.dilated_conv3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                    self.relu)
        # self.dilated_conv4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
        #                                    self.relu)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        # self.conv4 = nn.Sequential(self.encoder[17],
        #                            self.relu,
        #                            self.encoder[19],
        #                            self.relu,
        #                            self.encoder[21],
        #                            self.relu)
        # # self.aspp = ASPP(in_channels=256, out_channels=256)
        # self.conv5 = nn.Sequential(self.encoder[24],
        #                            self.relu,
        #                            self.encoder[26],
        #                            self.relu,
        #                            self.encoder[28],
        #                            self.relu)



        self.center = DecoderBlockV2(256, 256, 128, is_deconv)

        self.se_module_diff1 = Se_module_diff(inp=64, oup=16)
        self.se_module_diff2 = Se_module_diff(inp=128, oup=32)
        self.se_module_diff3 = Se_module_diff(inp=256, oup=64)
        # self.se_module_diff4 = Se_module_diff(inp=512, oup=128)
        # self.se_module_diff5 = Se_module_diff(inp=512, oup=256)
        """dec5"""
        # self.conv5_s = conv1x1(512, 256)
        # self.dec5 = DecoderBlockV2(256 * 2, 256 * 2, 256, is_deconv)
        #
        # """dec4"""
        # self.dec5_s = conv1x1(256, 128)
        # self.conv4_s = conv1x1(512, 128)
        # self.center_e = Interpolate(scale_factor=2, mode='bilinear')
        # self.center_s = conv1x1(256, 128)
        # self.dec4 = DecoderBlockV2(128 * 3, 128 * 2, 128, is_deconv)
        # self.deconv = nn.Sequential(self.dec5_s,
        #                             self.conv4_s)

        """dec3"""
        self.dec5_e1 = Interpolate(scale_factor=2, mode='bilinear')
        self.dec5_s1 = conv1x1(128, 64)
        self.dec4_s1 = conv1x1(128, 64)
        self.conv3_s = conv1x1(256, 64)
        self.center_e1 = Interpolate(scale_factor=2, mode='bilinear')
        self.center_s1 = conv1x1(128,64)
        self.dec3 = DecoderBlockV2(64 * 2, 64 * 2, 64, is_deconv)

        """dec2"""
        # self.dec5_e2 = Interpolate(scale_factor=2, mode='bilinear')
        # self.dec5_s2 = conv1x1(64, 32)
        # self.dec4_e2 = Interpolate(scale_factor=2, mode='bilinear')
        # self.dec4_s2 = conv1x1(64, 32)
        self.dec3_s2 = conv1x1(64, 32)
        self.conv2_s = conv1x1(128, 32)
        self.center_e2 = Interpolate(scale_factor=2, mode='bilinear')
        self.center_s2 = conv1x1(64, 32)
        self.dec2 = DecoderBlockV2(32 * 3, 32 * 3, 32, is_deconv)

        """dec1"""
        # self.dec5_e3 = Interpolate(scale_factor=2, mode='bilinear')
        # self.dec5_s3 = conv1x1(32, 16)
        # self.dec4_e3 = Interpolate(scale_factor=2, mode='bilinear')
        # self.dec4_s3 = conv1x1(32, 16)
        self.dec3_e3 = Interpolate(scale_factor=2, mode='bilinear')
        self.dec3_s3 = conv1x1(32, 16)
        self.dec2_s3 = conv1x1(32, 16)
        self.conv1_s = conv1x1(64, 16)
        self.center_e3 = Interpolate(scale_factor=2, mode='bilinear')
        self.center_s3 = conv1x1(32, 16)
        self.dec1 = ConvRelu(16 * 4, num_filters)


        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)                                                                      #64
        conv2 = self.conv2(self.dilated_conv1(conv1))                                                       #128
        conv3 = self.conv3(self.dilated_conv2(conv2))                                                       #256
        conv4 = self.conv4(self.dilated_conv3(conv3))                                                       #512
        conv5 = self.conv5(self.dilated_conv4(conv4))                                                       #512




        center = self.center(self.dilated_conv3(conv5))                                                   #256

        # """dec5"""
        # conv5_s = self.conv5_s(conv5)
        # conv5_sse = self.se_module_diff5(conv5, conv5_s)
        # dec5 = self.dec5(torch.cat([center, conv5_sse], 1))  # 256
        #
        # """dec4"""
        # dec5_s = self.dec5_s(dec5)
        # conv4_s = self.conv4_s(conv4)
        # conv4_sse = self.se_module_diff4(conv4, conv4_s)
        # center_e = self.center_e(center)
        # center_s = self.center_s(center_e)
        # dec4 = self.dec4(torch.cat([dec5_s, conv4_sse, center_s], 1))  # 128

        """dec3"""
        # dec5_e1 = self.dec5_e1(dec5_s)
        # dec5_s1 = self.dec5_s1(dec5_e1)
        # dec4_s1 = self.dec4_s1(dec4)
        conv3_s = self.conv3_s(conv3)
        conv3_sse = self.se_module_diff3(conv3, conv3_s)
        center_e1 = self.center_e1(center)
        center_s1 = self.center_s1(center_e1)
        dec3 = self.dec3(torch.cat([conv3_sse, center_s1], 1))  # 64

        """dec2"""
        # dec5_e2 = self.dec5_e2(dec5_s1)
        # dec5_s2 = self.dec5_s2(dec5_e2)
        # dec4_e2 = self.dec5_e2(dec4_s1)
        # dec4_s2 = self.dec4_s2(dec4_e2)
        dec3_s2 = self.dec3_s2(dec3)
        conv2_s = self.conv2_s(conv2)
        conv2_sse = self.se_module_diff2(conv2, conv2_s)
        center_e2 = self.center_e2(center_s1)
        center_s2 = self.center_s2(center_e2)
        dec2 = self.dec2(torch.cat([dec3_s2, conv2_sse, center_s2], 1))  # 32

        """dec1"""
        # dec5_e3 = self.dec5_e3(dec5_s2)
        # dec5_s3 = self.dec5_s3(dec5_e3)
        # dec4_e3 = self.dec4_e3(dec4_s2)
        # dec4_s3 = self.dec4_s3(dec4_e3)
        dec3_e3 = self.dec3_e3(dec3_s2)
        dec3_s3 = self.dec3_s3(dec3_e3)
        dec2_s3 = self.dec2_s3(dec2)
        conv1_s = self.conv1_s(conv1)
        conv1_sse = self.se_module_diff1(conv1, conv1_s)
        center_e3 = self.center_e3(center_s2)
        center_s3 = self.center_s3(center_e3)
        dec1 = self.dec1(torch.cat([dec3_s3, dec2_s3, conv1_sse, center_s3], 1))                       # 32

        # sse1 = self.se_module_diff1(conv1, dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)                                                                #32/1
            #x_out = F.sigmoid(x_out)

        return x_out

