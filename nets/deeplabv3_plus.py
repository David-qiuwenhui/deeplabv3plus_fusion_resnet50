import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.hrnet import HRNet_Backbone, hrnet_classification
from nets.hrnet_new import HRNet_Backbone_New
from nets.mobilenetv3 import mobilenet_v3_large_backbone
from nets.mobilevit import mobile_vit_small_backbone
from nets.repvgg_new import repvgg_backbone_new, repvgg_model_convert
from nets.resnet import resnet50_backbone
from nets.resnext import resnext50_32x4d_backbone
from nets.swin_transformer import Swin_Transformer_Backbone
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
    def __init__(
        self,
        num_classes,
        backbone,
        pretrained=False,
        downsample_factor=8,
        backbone_path="",
    ):
        super(DeepLab, self).__init__()
        self.backbone_name = backbone
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
        elif backbone == "resnet50":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [2048,H/8,W/8]
            #   浅层特征    [256,H/4,W/4]
            # ----------------------------------#
            self.backbone = resnet50_backbone(pretrained, backbone_path)
            in_channels = 2048  # 主干部分的特征
            low_level_channels = 256  # 浅层次特征

        elif backbone == "resnext50":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [2048,H/8,W/8]
            #   浅层特征    [256,H/4,W/4]
            # ----------------------------------#
            self.backbone = resnext50_32x4d_backbone(
                pretrained=False, downsample_factor=8
            )

            in_channels = 2048  # 主干部分的特征
            low_level_channels = 256  # 浅层次特征

        elif backbone == "repvgg_new":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [2560,H/8,W/8]
            #   浅层特征    [160,H/4,W/4]
            # ----------------------------------#
            self.backbone = repvgg_backbone_new(model_type="RepVGG-B2g4-new")
            in_channels = 2560  # 主干部分的特征
            low_level_channels = 160  # 浅层次特征

        elif backbone == "hrnet":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [480,H/8,W/8]
            #   浅层特征    [256,H/4,W/4]
            # ----------------------------------#
            self.backbone = HRNet_Backbone(
                backbone="hrnetv2_w32", pretrained=pretrained
            )
            in_channels = 480  # 主干部分的特征
            low_level_channels = 256  # 浅层次特征

        elif backbone == "hrnet_new":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [32,H/4,W/4]
            #   浅层特征    [256,H/4,W/4]
            #   注意：hrnet_new 的深浅层次融合特征尺寸是相同的
            # ----------------------------------#
            self.backbone = HRNet_Backbone_New(model_type="hrnet_w32")
            in_channels = 32  # 主干部分的特征
            low_level_channels = 256  # 浅层次特征

        elif backbone == "swin_transformer":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [1024,H/8,W/8]
            #   浅层特征    [256,H/4,W/4]
            # ----------------------------------#
            self.backbone = Swin_Transformer_Backbone()
            in_channels = 1024
            low_level_channels = 256

        elif backbone == "mobilevit":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [640,H/8,W/8]
            #   浅层特征    [64,H/4,W/4]
            # ----------------------------------#
            self.backbone = mobile_vit_small_backbone(model_type="small")
            in_channels = 640
            low_level_channels = 64

        elif backbone == "mobilenetv3":
            # ----------------------------------#
            #   获得两个特征层
            #   主干部分    [640,H/8,W/8]
            #   浅层特征    [64,H/4,W/4]
            # ----------------------------------#
            self.backbone = mobilenet_v3_large_backbone(model_type="large")
            in_channels = 160
            low_level_channels = 40
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
            nn.Conv2d(
                in_channels=low_level_channels,
                out_channels=256,  # deeplabv3plus 48
                kernel_size=1,
            ),
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
        H, W = x.size(2), x.size(3)  # x(bs,3,H,W)
        # -----------------------------------------#
        #   特征提取 获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理 (B, 256, H/4, W/4)  处理4倍下采样feature maps
        #   x : 主干部分-利用ASPP结构进行加强特征提取 (B, 2048, H/8, W/8)  处理8倍下采样feature maps
        # -----------------------------------------#

        if self.backbone_name in [
            "xception",
            "mobilenet",
            "repvgg_new",
            "hrnet",
            "swin_transformer",
            "mobilevit",
            "mobilenetv3",
            "hrnet_new",
        ]:
            low_level_features, x = self.backbone(x)
        elif self.backbone_name in ["resnet50", "resnext50"]:
            features = self.backbone(x)
            low_level_features = features["low_features"]  # (B, 256, H/4, W/4)
            x = features["main"]  # (B, 2048, H/8, W/8)

        x = self.aspp(x)  # x(bs, 256, H/8, W/8)
        low_level_features = self.shortcut_conv(
            low_level_features
        )  # low_level_features(bs, 256, H/4, W/4)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(
            input=x,
            size=(low_level_features.size(2), low_level_features.size(3)),
            mode="bilinear",
            align_corners=True,
        )  # x(bs, 256, H/8, W/8) -> x(bs, 256, H/4, W/4)
        x = self.cat_conv(
            torch.cat((x, low_level_features), dim=1)  # (bs,304,H/4,W/4)
        )  # x(bs, 256, H/4, W/4)
        x = self.cls_conv(x)  # x(bs, num_classes, H/4, W/4)
        x = F.interpolate(
            input=x, size=(H, W), mode="bilinear", align_corners=True
        )  # x(bs, num_classes, H, W)
        return x

    def switch_to_deploy(self):
        if self.backbone_name in ["repvgg_new"]:
            self.backbone = repvgg_model_convert(model=self.backbone)
            print(
                f"\033[1;33;44m 🔬🔬🔬🔬 Switch {self.backbone_name} to deploy model \033[0m"
            )
        else:
            print(f"\033[1;31;41m 🔬🔬🔬🔬 Can not Switch to deploy model \033[0m")
