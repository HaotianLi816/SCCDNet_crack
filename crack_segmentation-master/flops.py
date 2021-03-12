from thop import profile
import torch
import argparse
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
from thop import clever_format



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, default='C:\\Users\\Administrator\\Desktop\\DeepCrack\\0\\model_best.pt',
                        help='trained model path')
    parser.add_argument('-model_type', type=str, default='vgg16', choices=['vgg16', 'resnet101', 'resnet34'])
    args = parser.parse_args()
    if args.model_type == 'vgg16':
        model = load_unet_vgg16(args.model_path)
    elif args.model_type  == 'resnet101':
        model = load_unet_resnet_101(args.model_path)
    elif args.model_type  == 'resnet34':
        model = load_unet_resnet_34(args.model_path)
        print(model)
    else:
        print('undefind model name pattern')
        exit()
    model = model.cuda()
    input = torch.randn(1, 3, 224, 224)  # 模型输入的形状,batch_size=1
    input = input.cuda()



    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops , params )  # flops单位G，para单位M








