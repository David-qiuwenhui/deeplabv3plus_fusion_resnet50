import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True, downsample_factor=8):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


# -----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(
        self, dim_in, dim_out, rate=1, bn_mom=0.1
    ):  # dim_in=2048, dim_out=256, rate=2
        super(ASPP, self).__init__()
        # Conv1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv3x3 branch dilation=6 * 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=6 * rate,
                dilation=6 * rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv3x3 branch dilation=12 * 2
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=12 * rate,
                dilation=12 * rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv3x3 branch dilation=18 * 2
        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=18 * rate,
                dilation=18 * rate,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Conv1x1 branch 全局平均池化层
        self.branch5_conv = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.branch5_bn = nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        # 对ASPP模块concat后的结果进行卷积操作（降低维度）
        self.conv_cat = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_out * 5,
                out_channels=dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()  # x(bs,2048,64,64)
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)  # conv1x1(bs, 256, 64, 64)
        conv3x3_1 = self.branch2(x)  # conv3x3_1(bs, 256, 64, 64)
        conv3x3_2 = self.branch3(x)  # conv3x3_2(bs, 256, 64, 64)
        conv3x3_3 = self.branch4(x)  # conv3x3_3(bs, 256, 64, 64)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(
            input=x, dim=2, keepdim=True
        )  # global_feature(bs, 2048, 1, 64)
        global_feature = torch.mean(
            input=global_feature, dim=3, keepdim=True
        )  # global_feature(bs, 2048, 1, 1)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(
            global_feature
        )  # global_feature(bs, 256, 1, 1)
        global_feature = F.interpolate(
            input=global_feature,
            size=(row, col),
            scale_factor=None,
            mode="bilinear",
            align_corners=True,
        )  # global_feature(bs, 256, 64, 64)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征
        # -----------------------------------------#
        feature_cat = torch.cat(
            [conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1
        )  # feature_cat(bs, 1280, 64, 64)
        result = self.conv_cat(feature_cat)  # result(bs, 256, 64, 64)
        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone, pretrained=False, downsample_factor=8):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(pretrained, downsample_factor)
            in_channels = 2048  # 主干部分的特征 (2048,30,30)
            low_level_channels = 256  # 浅层特征 (256,128,128)
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(pretrained, downsample_factor)
            in_channels = 320  # 主干部分的特征(320,30,30)
            low_level_channels = 24  # 浅层特征(24,128,128)
        else:
            raise ValueError(
                "Unsupported backbone - `{}`, Use mobilenet, xception.".format(backbone)
            )

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = ASPP(
            dim_in=in_channels, dim_out=256, rate=16 // downsample_factor
        )  # dim_in=2048 dim_out=256 rate=2

        # ----------------------------------#
        #   浅层特征边的卷积处理模块 将通道维度调整为48
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(in_channels=low_level_channels, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
        )

        # Concat拼接浅层特征和ASPP处理后的特征
        self.cat_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        # 更改channels至num_classes
        self.cls_conv = nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1, stride=1
        )

    def forward(self, x):
        H, W = x.size(2), x.size(3)  # x(bs,3,512,512)
        # -----------------------------------------#
        #   特征提取 获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理 (bs, 256, 128, 128)  处理4倍下采样feature maps
        #   x : 主干部分-利用ASPP结构进行加强特征提取 (bs, 2048, 64, 64)  处理8倍下采样feature maps
        # -----------------------------------------#
        low_level_features, x = self.backbone(
            x
        )  # low_level_features(bs, 256, 128, 128)  x(bs, 2048, 64, 64)
        x = self.aspp(x)  # x(bs, 256, 64, 64)
        low_level_features = self.shortcut_conv(
            low_level_features
        )  # low_level_features(bs, 256, 128, 128)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(
            input=x,
            size=(low_level_features.size(2), low_level_features.size(3)),
            mode="bilinear",
            align_corners=True,
        )  # x(bs, 256, 64, 64) -> x(bs, 256, 128, 128)
        x = self.cat_conv(
            torch.cat((x, low_level_features), dim=1)  # (bs,304,128,128)
        )  # x(bs, 256, 128, 128)
        x = self.cls_conv(x)  # x(bs, num_classes, 128, 128)
        x = F.interpolate(
            input=x, size=(H, W), mode="bilinear", align_corners=True
        )  # x(bs, num_classes, H, W)
        return x
