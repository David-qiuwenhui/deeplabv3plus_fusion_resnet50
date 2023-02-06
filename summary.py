import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.deeplabv3_plus import DeepLab
from nets.repvgg_new import repvgg_model_convert


# --------------------------------------------#
#  模型结构的配置
#  计算模型的参数量和单位时间的浮点运算量
# --------------------------------------------#
model_cfg = dict(
    input_shape=[512, 512],
    num_classes=7,
    # xception, mobilenet, resnet50, resnext50, repvgg_new
    # hrnet, hrnet_new, swin_transformer, mobilevit, mobilenetv3
    backbone="hrnet_new",
    downsample_factor=8,
    deploy=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


def main(model_cfg):
    input_shape = model_cfg["input_shape"]
    num_classes = model_cfg["num_classes"]
    backbone = model_cfg["backbone"]
    device = model_cfg["device"]
    downsample_factor = model_cfg["downsample_factor"]
    deploy = model_cfg["deploy"]

    # ---------- 实例化深度卷积模型 ----------
    model = DeepLab(
        num_classes,
        backbone,
        pretrained=False,
        downsample_factor=downsample_factor,
    ).to(device)
    if deploy:
        if backbone in ["repvgg_new"]:
            model = repvgg_model_convert(model)

    # summary(model, input_size=(3, input_shape[0], input_shape[1]))

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input,), verbose=False)
    # --------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    # --------------------------------------------------------#
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print("Total GFLOPS: %s" % (flops))
    print("Total params: %s" % (params))


if __name__ == "__main__":
    main(model_cfg)
