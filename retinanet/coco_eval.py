import os

import numpy as np
from pycocotools.cocoeval import COCOeval
import json
import torch
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


def evaluate_coco(dataset, model, threshold=0.05, type='val', save_epoch=0,save_path='results'):  # save_path 评估结果保存路径
    model.eval()

    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                # for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id': dataset.image_ids[index],
                        'category_id': dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open(os.path.join(save_path, '{}_bbox_results.json'.format(dataset.set_name)), 'w'),
                  indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(os.path.join(save_path, '{}_bbox_results.json'.format(dataset.set_name)))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        # coco_eval.params.catIds = [3]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 保存训练时的评估结果
        # if type == 'train':
        #     with open('eval_summary.txt', 'a') as f:
        #         f.write(str(save_epoch)+':'+'\n')
        #         summ = coco_eval.summarize()
        #         f.write(str(summ))
        #         f.write('\n\n')

        # 当训练状态时平均精度与Epoch的关系
        if type == 'train':
            coco_eval.restore_AP_Epoch(save_epoch, 'results')

        if type == 'val':
            plot_(coco_true, coco_eval, save_path)
        # calculate(coco_true, coco_eval)
        model.train()

        return


# def plot_(coco, coco_eval, image_ids, save_path):
#     cats = coco.loadCats(coco_eval.params.catIds)
#     catIds = coco_eval.params.catIds
#     coco_eval.params.imgIds = image_ids
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     precisions = coco_eval.eval['precision']
#     recalls = coco_eval.eval['recall']
#     nms = [cat['name'] for cat in cats]
#
#     # print(precisions, recalls)
#
#     for idx, catId in enumerate(catIds):
#         pre = precisions[0, :, idx, 0, 2]
#         x = np.arange(0.0, 1.01, 0.01)
#         plt.figure(figsize=(7, 5))
#         # plt.plot(recalls[idx], precisions[idx])
#         # 画AP
#         plot_ap(nms[idx], x, pre, save_path)
#
#     # 针对不同的IOU阈值计算结果 f1,prec,rec
#     # 对于不同类别计算结果
#     for idx, cat in enumerate(catIds):
#         f1_list = []
#         prec_list = []
#         rec_list = []
#         for iou in np.arange(0.5, 1.0, 0.05):
#             coco_eval.params.iouThrs = [iou]  # 设置IOU阈值
#             coco_eval.params.catIds = [idx + 1]  # 设置类别
#
#             # 计算F1、precision、recall
#             coco_eval.computeF1(coco_eval.params.catIds[0])
#             prec = precisions[0, :, idx, iou]
#             rec = recalls[0, :, idx, iou]
#             f1 = 2 * prec * rec / (prec + rec)
#
#             prec_list.append(prec)
#             rec_list.append(rec)
#             f1_list.append(f1)
#
#         # 画f1,precision,recall
#         plot_fpr(np.arange(0.5, 1.0, 0.05), prec_list, 'IOU', 'Precision', nms[idx], 'Precision', save_path)
#         plot_fpr(np.arange(0.5, 1.0, 0.05), rec_list, 'IOU', 'Recall', nms[idx], 'Recall', save_path)
#         plot_fpr(np.arange(0.5, 1.0, 0.05), f1_list, 'IOU', 'F1', nms[idx], 'F1', save_path)


# 画AP
def plot_(coco, coco_eval, save_path):
    cats = coco.loadCats(coco_eval.params.catIds)
    catIds = coco_eval.params.catIds
    # coco_eval.params.imgIds = image_ids
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    precisions = coco_eval.eval['precision']
    recalls = coco_eval.eval['recall']
    nms = [cat['name'] for cat in cats]

    # print(precisions, recalls)

    for idx, catId in enumerate(catIds):
        pre = precisions[0, :, idx, 0, 2]
        x = np.arange(0.0, 1.01, 0.01)
        plt.figure(figsize=(7, 5))
        # 画AP
        plot_ap(nms[idx], x, pre, save_path)

    results_per_category = []

    total_f1 = []
    results_per_category_iou50 = []
    results_per_category_iou75 = []
    # 针对不同的IOU阈值计算结果 f1,prec,rec
    # 对于不同类别计算结果
    for idx, cat in enumerate(catIds):
        # 计算F1、precision、recall
        # coco_eval.computeF1(coco_eval.params.catIds[0])
        precision = precisions[:, :, idx, 0, -1]
        precision_50 = precisions[0, :, idx, 0, -1]
        precision_75 = precisions[1, :, idx, 0, -1]
        precision = precision[precision > -1]

        recall = recalls[:, idx, 0, -1]
        recall_50 = recalls[0, idx, 0, -1]
        recall_75 = recalls[1, idx, 0, -1]
        recall = recall[recall > -1]

        if precision.size:
            ap = np.mean(precision)
            ap_50 = np.mean(precision_50)
            ap_75 = np.mean(precision_75)
            rec = np.mean(recall)
            rec_50 = np.mean(recall_50)
            rec_75 = np.mean(recall_75)
        else:
            ap = float('nan')
            ap_50 = float('nan')
            ap_75 = float('nan')
            rec = float('nan')
            rec_50 = float('nan')
            rec_75 = float('nan')

        # 保存每个类别的ap和rec
        res_item = [f'{nms[idx]}', f'{float(ap):0.3f}', f'{float(rec):0.3f}']
        results_per_category.append(res_item)

        res_item_50 = [f'{nms[idx]}', f'{float(ap_50):0.3f}', f'{float(rec_50):0.3f}']
        f1_score = 2 * float(res_item[1]) * float(res_item[2]) / (
                float(res_item[1]) + float(res_item[2]) + 1e-6)

        res_item_75 = [f'{nms[idx]}', f'{float(ap_75):0.3f}', f'{float(rec_75):0.3f}']

        total_f1.append(f1_score)
        print("种类:{0},ap(.50:.95 .50 .75):{1} {2} {3},r:{4} {5} {6},f1 score:{7}".format(res_item[0],
                                                                                           res_item[1],
                                                                                           res_item_50[1],
                                                                                           res_item_75[1],
                                                                                           res_item[2],
                                                                                           res_item_50[2],
                                                                                           res_item_75[2],
                                                                                           f1_score))

        results_per_category_iou50.append(res_item_50)
    print('总体的f1_score 为:{}'.format(sum(total_f1) / len(total_f1)))


# 画AP
def plot_ap(name, x, y, save_path):
    plt.title(name)  # plt.title(nms[idx] + ' Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(os.path.join(save_path, 'AP_' + name + '_result.png'))


# 画F1、precision和recall随不同IOU的变化
def plot_fpr(x, y, x_label, y_label, name, plot_type, save_path):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(os.path.join(save_path, plot_type + '/' + name + '.png'))


### 暂时走不通
def calculate(coco, coco_eval):
    catIds = coco.getCatIds()
    coco_eval.params.catIds = catIds
    coco_eval.evaluate()
    coco_eval.accumulate()

    # 获取指标值
    AP = coco_eval.stats
    print(AP)
    f1s = []

    for idx, catId in enumerate(catIds):
        tp = coco_eval.eval['precision'][idx, :, :, catId - 1, 0, 1]
        fp = coco_eval.eval['precision'][idx, :, :, catId - 1, 0, 2]
        fn = coco_eval.eval['precision'][idx, :, :, catId - 1, 1, 0]
        f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
        f1s.append(f1.max())

    # 打印结果
    for idx, catId in enumerate(catIds):
        print('{} AP:{},F1:{}'.format(coco.loadCats(catId)[0]['name'],
                                      AP[idx], f1s[idx]))
