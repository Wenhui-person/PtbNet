import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 加载预训练的ResNet50模型
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)


# 定义CAM模型
class CAM(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(CAM, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.gradient = None

    def gradients_hook(self, grad):
        self.gradient = grad

    def forward(self, x):
        x = self.feature_extractor(x)
        h = x.register_hook(self.gradients_hook)
        x = self.classifier(x)
        h.remove()
        return x


# 定义CAM可视化函数
def visualize_CAM(image, model, target_class):
    # 转换图像为张量
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)

    # 定义CAM模型
    feature_extractor = nn.Sequential(*list(model.children())[:-2])
    classifier = model.fc
    cam_model = CAM(feature_extractor, classifier)

    # 前向传播
    output = cam_model(image)
    target = torch.Tensor([target_class])
    loss = nn.functional.binary_cross_entropy_with_logits(output, target)

    # 反向传播
    cam_model.zero_grad()
    loss.backward()

    # 计算CAM
    features = cam_model.feature_extractor(image).detach().squeeze()
    weights = cam_model.gradient.squeeze().mean(dim=(1, 2), keepdims=True)
    cam = (weights * features).sum(dim=0)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    # 可视化CAM
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.imshow(cam, cmap='jet', alpha=0.5, interpolation='nearest')
    plt.axis('off')
    plt.show()


# 加载一张测试图像
image = plt.imread('test_image.jpg')

# 可视化CAM
visualize_CAM(image, model, target_class=0)
