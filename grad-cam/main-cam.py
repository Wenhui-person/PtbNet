import os

import cv2
import numpy as np
import skimage
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img


def resizer(image, min_side=1500, max_side=3000):
    rows, cols, cns = image.shape

    # smallest_side = min(rows, cols)
    #
    # # rescale the image so the smallest side is min_side
    # scale = min_side / smallest_side
    #
    # # check if the largest side is now greater than max_side, which can happen
    # # when images have a large aspect ratio
    # largest_side = max(rows, cols)
    #
    # if largest_side * scale > max_side:
    #     scale = max_side / largest_side
    scale = 1/4
    # resize the image with the computed scale
    image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
    # print(image.shape)
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    return new_image


def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]
    #
    # # model = models.vgg16(pretrained=True)
    # # target_layers = [model.features]
    #
    # # model = models.resnet34(pretrained=True)
    # # target_layers = [model.layer4]
    #
    # # model = models.regnet_y_800mf(pretrained=True)
    # # target_layers = [model.trunk_output]
    #
    # # model = models.efficientnet_b0(pretrained=True)
    # # target_layers = [model.features]

    # data_transform = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # # load image
    # img_path = "both.png"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path).convert('RGB')
    # img = np.array(img, dtype=np.uint8)
    # # img = center_crop_img(img, 224)
    #
    # # [C, H, W]
    # img_tensor = data_transform(img)
    # # expand batch dimension
    # # [C, H, W] -> [N, C, H, W]
    # input_tensor = torch.unsqueeze(img_tensor, dim=0)

    image_path = r"E:\Yangwenhui\Projects\retinanet_new\images\20150519000884-11.jpg"
    model_path = r'E:\Yangwenhui\Projects\retinanet_new\logs\2023-08-18104919\weights\retinanet_225.pt'

    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = cv2.imread(image_path)
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    img = resizer(img)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    retinanet = torch.load(model_path).cuda()
    retinanet.eval()

    target_layers = [retinanet.module.classificationModel.conv4]

    cam = GradCAM(model=retinanet, target_layers=target_layers, use_cuda=False)
    target_category = 1

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(img.astype(dtype=np.float32),
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig('00884-cam_BiFPN_PSPD-Conv_组会.png', dpi=400, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
    # image_path = r"E:\Yangwenhui\Data\FJH2022\after_ddpm\images\20150511000352-11.jpg"
    # img = cv2.imread(image_path)
    # scale = 1/4
    # rows, cols, cns = img.shape
    # image = skimage.transform.resize(img, (int(round(rows * scale)), int(round((cols * scale)))))
    # print(image.shape)
    # rows, cols, cns = image.shape
    # pad_w = 32 - rows % 32
    # pad_h = 32 - cols % 32
    # new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    # new_image[:rows, :cols, :] = image.astype(np.float32)
    # print(new_image.shape)
    # cv2.imshow('prac', new_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
