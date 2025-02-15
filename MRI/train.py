# -*- encoding: utf-8 -*-
'''
@File    :   main_train.py
@Time    :   2022/10/17 18:03:54
@Author  :   Bercy 
@Version :   1.1
@Contact :   hebingxi0616@163.com
'''

import os
import torch
import pandas as pd
import torch.nn as nn
import argparse
from datasets import NeoadjuvantImmunotherapy_Dataset
from torch.utils.data import Dataset, DataLoader
from utils import get_result_binary, adjust_learning_rate
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
from models.resnext import resnext101
from models.mobilenet import MobileNet
from models.mobilenetv2 import MobileNetV2
from models.shufflenet import ShuffleNet
from models.shufflenetv2 import ShuffleNetV2
from models.squeezenet import SqueezeNet
from sklearn.metrics import confusion_matrix
from torch import optim
from monai.transforms import (
    Compose,
    ScaleIntensity,
    AddChannel,
    Resize,
    RandRotate90,
    RandFlip,
    RandZoom,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandAdjustContrast,
    EnsureType
)

import SimpleITK as sitk

class TrainTransforms:
    def __init__(self):
        self.transform = Compose([
            RandRotate90(prob=0.5),      # 随机旋转 90 度
            RandFlip(spatial_axis=0, prob=0.5),    # 沿第一个轴随机翻转
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5), # 随机缩放
            RandGaussianNoise(prob=0.5), # 添加高斯噪声
            RandGaussianSmooth(prob=0.5),# 随机高斯模糊
            RandAdjustContrast(prob=0.5),# 随机对比度调整
            # EnsureType()                 # 转换为 MONAI 的张量类型
        ])

    def __call__(self, img):
        return self.transform(img)

def load_and_preprocess(root_dir, image_size):
    datas = []
    labels = []
    imgs_path = []
    data_transforms = Compose([
            ScaleIntensity(),            # 将像素值缩放到 [0, 1] 范围
            AddChannel(),                # 增加通道维度，使其成为单通道图像
            Resize(image_size),            # 调整图像大小到指定的 roi_size
            EnsureType()  # 转换为 MONAI 的张量类型
        ])
    # 遍历文件夹中的图片，按文件夹名（0 或 1）设置标签
    for label in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, str(label))
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            img = data_transforms(img)  # 应用 transform（如 resize、归一化等）
            datas.append(img)
            labels.append(int(label))
            imgs_path.append(img_name)

    return datas, labels, imgs_path

class Dataset_transform(Dataset):
    def __init__(self, imgs, labels, paths, transform=None):
        """
        root_dir: 文件夹路径，例如 'train/' 或 'test/'，其下有子文件夹 '0' 和 '1'
        parameter: 标准化所需的均值和方差（mu, sigma）
        image_size: 目标图像大小，默认为 (128, 128, 128)
        transform: 图像大小调整函数
        standard: 标准化函数
        loader: 加载图像函数，默认使用 SimpleITK
        num_class: 分类数量，默认为 2
        """
        self.imgs = imgs
        self.labels = labels
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        path = self.paths[index]
        return img, label, path

    def __len__(self):
        return len(self.imgs)

def evaluate_model(model, loader, criterion, task_type, output_excel=None):
    model.eval()
    loss_sum = 0
    total = 0
    output_outputs = torch.Tensor([]).to(device)  # 模型输出（概率或 logits）
    output_pred = torch.Tensor([]).to(device)  # 预测类别
    labels = torch.IntTensor([]).to(device)  # 真实标签

    # 创建一个列表来存储每个批次的数据
    results_list = []

    for inputs, label, imgs_path in loader:
        inputs, label = inputs.float().to(device), label.unsqueeze(1).float().to(
            device) if task_type == 'binary' else label.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, label)
            total += 1
            loss_sum += loss.item()

            if task_type == 'binary':
                # 二分类任务下，计算概率和预测值
                probabilities = torch.sigmoid(outputs)
                predicted = torch.round(probabilities)  # 概率大于0.5即为正类
                output_outputs = torch.cat((output_outputs, probabilities.float()), 0)  # 模型的概率输出

            else:
                # 多分类任务下，使用 softmax 和 argmax 获取预测类别
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                output_outputs = torch.cat((output_outputs, probabilities[:, 1].float()), 0)  # 模型的概率输出

            output_pred = torch.cat((output_pred, predicted.float()), 0)  # 预测类别
            labels = torch.cat((labels, label.int()), 0)

            # 将当前批次的数据合并到结果列表中
            for i in range(len(imgs_path)):
                results_list.append({
                    'image_path': imgs_path[i],  # 图片路径
                    'label': label[i].item(),  # 真实标签
                    'predicted': predicted[i].item(),  # 预测标签
                    'probability': probabilities[i].cpu().numpy().tolist() if task_type == 'binary' else probabilities[
                        i].cpu().numpy().tolist(),  # 概率输出
                })

    loss_sum = round(loss_sum / total, 4)

    if task_type == 'binary':
        conf_matrix = confusion_matrix(labels.cpu(), output_pred.cpu(), labels=[0, 1])
    else:
        conf_matrix = confusion_matrix(labels.cpu(), output_pred.cpu())

    print(conf_matrix)
    results = get_result_binary(output_outputs.cpu(), output_pred.cpu(),
                                labels.cpu())  # if task_type == 'binary' else None

    # 将结果列表转换为 DataFrame
    df_results = pd.DataFrame(results_list)

    # 保存为 .xlsx 文件
    if output_excel is not None:
        df_results.to_excel(output_excel, index=False)

    return loss_sum, results


def build_model(parameters, source_path, outfolder, device, pretrained, model_name, model_weight, task_type, num_classes, patch_size=96, sample_duration=96):
    lr = parameters[0]
    weight_decay = parameters[1]
    batch_size = parameters[2]
    EPOCHS = parameters[3]

    os.makedirs(outfolder, exist_ok=True)

    # 根据模型名称加载模型
    if model_name == 'Resnet18':
        model = resnet18(num_classes=num_classes, sample_size=patch_size, sample_duration=sample_duration, shortcut_type='A').to(device)
    elif model_name == 'Resnet34':
        model = resnet34(num_classes=num_classes, sample_size=patch_size, sample_duration=sample_duration, shortcut_type='A').to(device)
    elif model_name == 'Resnet50':
        model = resnet50(num_classes=num_classes, sample_size=patch_size, sample_duration=sample_duration).to(device)
    elif model_name == 'Resnet101':
        model = resnet101(num_classes=num_classes, sample_size=patch_size, sample_duration=sample_duration).to(device)
    elif model_name == 'Resnext101':
        model = resnext101(num_classes=num_classes, sample_size=patch_size, sample_duration=sample_duration).to(device)
    elif model_name == 'Resnet152':
        model = resnet152(num_classes=num_classes, sample_size=patch_size, sample_duration=sample_duration).to(device)
    elif model_name == 'Resnet200':
        model = resnet200(num_classes=num_classes, sample_size=patch_size, sample_duration=sample_duration).to(device)
    elif model_name == 'ShuffleNetV2':
        model = ShuffleNetV2(num_classes=num_classes, sample_size=patch_size, width_mult=2.).to(device)
    elif model_name == 'SqueezeNet':
        model = SqueezeNet(num_classes=num_classes, sample_size=patch_size, sample_duration=sample_duration).to(device)
    else:
        raise ValueError(f"Unknown model name {model_name}")

    resize_size = (patch_size, patch_size, sample_duration)
    train_path = os.path.join(source_path, 'train')
    valid_path = os.path.join(source_path, 'test')

    train_imgs_data, train_labels_data, train_imgs_path = load_and_preprocess(train_path, resize_size)
    print(f'train num: {len(train_imgs_data)}')
    # train_dataset = Dataset_transform(imgs=train_imgs_data, labels=train_labels_data, transform=TrainTransforms())
    train_dataset = Dataset_transform(imgs=train_imgs_data, labels=train_labels_data, paths=train_imgs_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_imgs_data, valid_labels_data, valid_imgs_path = load_and_preprocess(valid_path, resize_size)
    print(f'valid num: {len(valid_imgs_data)}')
    valid_dataset = Dataset_transform(imgs=valid_imgs_data, labels=valid_labels_data, paths=valid_imgs_path)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    # 根据任务类型选择损失函数
    if task_type == 'binary':
        positive_count, negative_count = 0, 0
        for _, label, _ in train_loader:
            positive_count += (label == 1).sum().item()
            negative_count += (label == 0).sum().item()
        pos_weight = torch.tensor([negative_count / positive_count]).to(device)
        print(f"Calculated pos_weight: {pos_weight}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    print(model)
    if pretrained and model_weight is not None:
        pretrained_dict = torch.load(model_weight)
        model_dict = model.state_dict()
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('加载预训练权重成功')

    # 训练过程
    train_loss, valid_loss = [], []
    train_metrics, valid_metrics = [], []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        # lr = adjust_learning_rate(optimizer, epoch, lr, 1e-3)
        lr = scheduler.get_last_lr()
        sum_loss = 0
        scheduler.step()
        for inputs, label, _ in train_loader:
            inputs, label = inputs.float().to(device), label.unsqueeze(1).float().to(device) if task_type == 'binary' else label.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

        # 评估模型
        train_loss_sum, train_result = evaluate_model(model, train_loader, criterion, task_type,)
        out_valid_list = os.path.join(outfolder, 'valid_list.xlsx')
        valid_loss_sum, valid_result = evaluate_model(model, valid_loader, criterion, task_type, output_excel=out_valid_list)

        train_loss.append(train_loss_sum)
        valid_loss.append(valid_loss_sum)

        if task_type == 'binary':
            train_metrics.append({
                'accuracy': train_result[0],
                'auc': train_result[1],
                'precision': train_result[2],
                'recall': train_result[3],
                'f1': train_result[4]
            })

            valid_metrics.append({
                'accuracy': valid_result[0],
                'auc': valid_result[1],
                'precision': valid_result[2],
                'recall': valid_result[3],
                'f1': valid_result[4]
            })

        torch.save(model.state_dict(), f'{outfolder}/net_{epoch:03d}.pth')

        # print(f'Epoch: {epoch} | Train Loss: {train_loss_sum:.6f} | '
        #       f'Train Acc: {100 * train_result[0]:.3f}% | Train AUC: {100 * train_result[1]:.3f}% | '
        #       f'Train Precision: {100 * train_result[2]:.3f}% | Train Recall: {100 * train_result[3]:.3f}% | '
        #       f'Train F1: {100 * train_result[4]:.3f}% | lr: {lr:.4e}')

        print(f'Epoch: {epoch} | Valid Loss: {valid_loss_sum:.6f} | '
              f'Valid Acc: {100 * valid_result[0]:.3f}% | Valid AUC: {100 * valid_result[1]:.3f}% | '
              f'Valid Precision: {100 * valid_result[2]:.3f}% | Valid Recall: {100 * valid_result[3]:.3f}% | '
              f'Valid F1: {100 * valid_result[4]:.3f}% | lr: {lr[0]:.4e}')

    train_df = pd.DataFrame(train_metrics, index=range(1, EPOCHS + 1))
    train_df['loss'] = train_loss
    valid_df = pd.DataFrame(valid_metrics, index=range(1, EPOCHS + 1))
    valid_df['loss'] = valid_loss

    return train_df, valid_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Model for Neoadjuvant Immunotherapy Dataset")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--model_name', type=str, default='Resnet101', help='Model name')
    parser.add_argument('--model_weight', type=str, default='weights/MedicalNet_pytorch_files2/pretrain/resnet_101.pth', help='Pretrained model weight')
    parser.add_argument('--outfolder', type=str, default='resnet50_results/fold_0',  help='Output folder to save models and logs')
    parser.add_argument('--source_path', type=str, default=r'/mnt/e/skx/Breast-Ultrasound/Datasets/MRI-t1/cut/no_background_50/1/4folds/fold_0', help='Source path for the dataset')
    parser.add_argument('--pretrained', default=True, help='Use pretrained weights')
    parser.add_argument('--task_type', type=str, choices=['binary', 'multiclass'], default='binary', help='Task type: binary or multiclass')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes for classification')

    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parameters = [args.lr, args.wd, args.batch_size, args.epochs]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df, valid_df = build_model(parameters, args.source_path, args.outfolder, device, args.pretrained, args.model_name, args.model_weight, args.task_type, args.num_classes)

    train_df.to_csv(os.path.join(args.outfolder, 'train_results.csv'))
    valid_df.to_csv(os.path.join(args.outfolder, 'valid_results.csv'))
