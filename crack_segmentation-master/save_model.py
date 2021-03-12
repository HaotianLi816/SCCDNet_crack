import torch
import onnx
import torch
import torchvision
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34

# model = torch.load("C:\\Users\\Administrator\\Desktop\\model_best.pt")
model = load_unet_vgg16("C:\\Users\\Administrator\\Desktop\\model_best.pt")
# model = torchvision.models.resnet18(pretrained=True)
# model.eval()
example = torch.rand(1, 3, 448, 448)
# traced_script_module = torch.jit.trace(model, example)



torch_out = torch.onnx._export(model.cuda(),             # model being run
                               example.cuda(),                       # model input (or a tuple for multiple inputs)
                                "C:\\Users\\Administrator\\Desktop\\sccdnet.onnx",
                               export_params = True,
                               opset_version = 11,
                               do_constant_folding = True,
)      # store the trained parameter weights inside the model file