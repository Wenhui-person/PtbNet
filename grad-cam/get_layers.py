import torch
from torch.utils import model_zoo

model_path = r'E:\Yangwenhui\Projects\retinanet_new\resnet50-19c8e357.pth'
retinanet = torch.load(model_path)
# print(retinanet.module)

for k,v in retinanet.items():
    print(k)
# for name, module in retinanet.module._modules.items():
#     print(name,module)

## 查看每层的名称
# for name in retinanet.state_dict():
#   print(name)
# # 输出相应层的权重
# print(retinanet.state_dict()['module.fn.2.conv3_up.bn.bias'])
# 打印模块的名字和参数大小
# for name, parameters in retinanet.named_parameters():
#     print(name, ';', parameters.size())
# 打印模块结构和模块名
# for name, module in retinanet.named_modules():
#     print(name)
