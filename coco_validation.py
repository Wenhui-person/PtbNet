import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory', default=r"E:\Yangwenhui\Data\FJH2022\Aug_ml")
    parser.add_argument('--model_path', help='Path to model', type=str,
                        default=r'E:\Yangwenhui\Projects\retinanet_new\logs\2023-08-18104919\weights\retinanet_175.pt')
    parser.add_argument('--sava_eval_path', help='Path to eval.', type=str,
                        default='logs/2023-08-18104919/evaluation_results')

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='val_test',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    # retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)

    retinanet = torch.load(parser.model_path)
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        # retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    # retinanet.module.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet, save_path=parser.sava_eval_path)


if __name__ == '__main__':
    main()
