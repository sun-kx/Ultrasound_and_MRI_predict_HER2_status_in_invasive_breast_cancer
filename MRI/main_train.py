# -*- encoding: utf-8 -*-
'''
@File    :   main_train.py
@Time    :   2022/10/17 18:03:54
@Author  :   Bercy 
@Version :   1.0
@Contact :   hebingxi0616@163.com
'''

import os
import torch
import pandas as pd
import torch.nn as nn
import argparse
from datasets import NeoadjuvantImmunotherapy_Dataset
from torch.utils.data import DataLoader
from utils import get_result_binary, adjust_learning_rate
from models.resnet import resnet18, resnet50, resnet101, resnet152
from models.resnext import resnext101
from models.mobilenet import MobileNet
from models.mobilenetv2 import MobileNetV2
from models.shufflenet import ShuffleNet
from models.shufflenetv2 import ShuffleNetV2
from models.squeezenet import SqueezeNet
from sklearn.metrics import confusion_matrix

def evaluate_model(model, loader, criterion, num_classes):
    model.eval()
    loss_sum = 0
    total = 0
    output_outputs = torch.Tensor([]).to(device)  # 模型输出（概率）
    output_pred = torch.Tensor([]).to(device)     # 预测类别
    labels = torch.IntTensor([]).to(device)       # 真实标签

    for inputs, label in loader:
        inputs, label = inputs.float().to(device), label.unsqueeze(1).float().to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, label)

            # 将 logits 转换为概率值
            probabilities = torch.sigmoid(outputs)

            # 根据概率值判断类别（概率大于0.5即为正类，等同于 sigmoid(outputs) > 0.5）
            predicted= torch.round(probabilities)
            # predicted = (probabilities > 0.5).float()
            # print(label.size(0))
            # total += label.size(0)
            total += 1
            # print(loss)
            loss_sum += loss.item()

            output_outputs = torch.cat((output_outputs, probabilities.float()), 0)  # 使用 sigmoid 后的概率
            output_pred = torch.cat((output_pred, predicted.float()), 0)           # 预测类别
            labels = torch.cat((labels, label.int()), 0)

    loss_sum = round(loss_sum / total, 4)

    # 计算混淆矩阵 (真实标签 vs 预测标签)
    conf_matrix = confusion_matrix(labels.cpu(), output_pred.cpu(), labels=[0, 1])
    print(conf_matrix)

    # 使用输出的概率、预测标签和真实标签计算二分类任务结果
    results = get_result_binary(output_outputs.cpu(), output_pred.cpu(), labels.cpu())

    return loss_sum, results



def build_model(parameters, source_path, outfolder, device, pretrained, model_name, model_weight, num_classes):

    lr = parameters[0]
    weight_decay = parameters[1]
    batch_size = parameters[2]
    EPOCHS = parameters[3]

    train_path = os.path.join(source_path, 'train')
    valid_path = os.path.join(source_path, 'test')

    os.makedirs(outfolder, exist_ok=True)

    if model_name == 'Resnet18':
        model = resnet18(num_classes=num_classes, sample_size=64,sample_duration=64).to(device)
    elif model_name == 'Resnet50':
        model = resnet50(num_classes=num_classes, sample_size=64,sample_duration=64).to(device)
    elif model_name == 'Resnet101':
        model = resnet101(num_classes=num_classes, sample_size=64,sample_duration=64).to(device)
    elif model_name == 'Resnext101':
        model = resnext101(num_classes=num_classes, sample_size=64,sample_duration=64).to(device)
    elif model_name == 'Resnet152':
        model = resnet152(num_classes=num_classes, sample_size=64, sample_duration=64).to(device)
    elif model_name == 'ShuffleNetV2':
        model = ShuffleNetV2(num_classes=num_classes, sample_size=64, width_mult=2.).to(device)

    else:
        raise ValueError(f"Unknown model name {model_name}")
    print(model)

    # 加载数据集以计算 pos_weight
    resize_size = (64, 64, 64)

    train_dataset = NeoadjuvantImmunotherapy_Dataset(root_dir=train_path, parameter=(0.5, 0.5), image_size=resize_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 计算 pos_weight
    positive_count = 0
    negative_count = 0
    for _, label in train_loader:
        positive_count += (label == 1).sum().item()
        negative_count += (label == 0).sum().item()

    pos_weight = torch.tensor([negative_count / positive_count]).to(device)
    print(f"Calculated pos_weight: {pos_weight}")

    # 使用 pos_weight 创建损失函数
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if pretrained and model_weight is not None:
        pretrained_dict = torch.load(model_weight)
        model_dict = model.state_dict()
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('加载预训练权重成功')

    valid_dataset = NeoadjuvantImmunotherapy_Dataset(root_dir=valid_path, parameter=(0.5, 0.5), image_size=resize_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    # 训练过程
    train_loss, valid_loss = [], []
    train_metrics, valid_metrics = [], []

    for epoch in range(1, EPOCHS + 1):
        lr = adjust_learning_rate(optimizer, epoch, lr, 1e-3)
        model.train()
        sum_loss = 0

        for inputs, label in train_loader:
            inputs, label = inputs.float().to(device), label.unsqueeze(1).float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

        # 评估模型
        train_loss_sum, train_result = evaluate_model(model, train_loader, criterion, num_classes)
        valid_loss_sum, valid_result = evaluate_model(model, valid_loader, criterion, num_classes)

        train_loss.append(train_loss_sum)
        valid_loss.append(valid_loss_sum)

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

        print(f'Epoch: {epoch} | Valid Loss: {valid_loss_sum:.6f} | '
              f'Valid Acc: {100 * valid_result[0]:.3f}% | Valid AUC: {100 * valid_result[1]:.3f}% | '
              f'Valid Precision: {100 * valid_result[2]:.3f}% | Valid Recall: {100 * valid_result[3]:.3f}% | '
              f'Valid F1: {100 * valid_result[4]:.3f}% | lr: {lr:.4e}')

    train_df = pd.DataFrame(train_metrics, index=range(1, EPOCHS + 1))
    train_df['loss'] = train_loss

    valid_df = pd.DataFrame(valid_metrics, index=range(1, EPOCHS + 1))
    valid_df['loss'] = valid_loss

    return train_df, valid_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Model for Neoadjuvant Immunotherapy")
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=2e-4, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--model', type=str, default='Resnext101', help='model name')
    parser.add_argument('--model_weight', type=str, default='weights/kinetics_resnext_101_RGB_16_best.pth', help='Path to pretrained weights')
    parser.add_argument('--num_classes', type=int, default=1, help='num classes')
    parser.add_argument('--pretrained', default=True,  help='Use pretrained weights')
    parser.add_argument('--source_path', type=str, default=r'E:\skx\Breast-Ultrasound\Datasets\MRI-t1\cut\no_background_50\5folds\fold_0', help='Path to training data table')
    parser.add_argument('--output', type=str, default='Resnet101_3e-5_results', help='output folder name')
    args = parser.parse_args()

    # 统一设置 device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 选择第一个 GPU (从 0 开始计数)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 根据系统是否有 GPU 选择

    parameters = [args.lr, args.wd, args.batch_size, args.epochs]

    # 开始模型训练
    train, valid = build_model(parameters, args.source_path, args.output, device, args.pretrained, args.model,
                               args.model_weight, args.num_classes)

    result_csv = pd.DataFrame({
        'train_loss': train['loss'], 'train_acc': train['accuracy'], 'train_auc': train['auc'], 'train_precision': train['precision'], 'train_recall': train['recall'], 'train_f1': train['f1'],
        'valid_loss': valid['loss'], 'valid_acc': valid['accuracy'], 'valid_auc': valid['auc'], 'valid_precision': valid['precision'], 'valid_recall': valid['recall'], 'valid_f1': valid['f1']
    })

    os.makedirs('result', exist_ok=True)
    result_csv.to_csv(os.path.join(args.output, f'{args.output}.csv'), index=False)

