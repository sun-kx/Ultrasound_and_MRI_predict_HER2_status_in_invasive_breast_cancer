import os
import torch
import pandas as pd
import argparse
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import confusion_matrix
from models.resnet import resnet18, resnet50, resnet101
from utils import get_result_binary
from monai.transforms import (
    Compose,
    ScaleIntensity,
    AddChannel,
    Resize,
    EnsureType
)
import SimpleITK as sitk


class Dataset_transform(Dataset):
    def __init__(self, imgs, labels, paths, transform=None):
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


def load_and_preprocess(root_dir, image_size):
    datas = []
    labels = []
    imgs_path = []
    data_transforms = Compose([
        ScaleIntensity(),
        AddChannel(),
        Resize(image_size),
        EnsureType()
    ])
    for label in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, str(label))
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            img = data_transforms(img)
            datas.append(img)
            labels.append(int(label))
            imgs_path.append(img_name)

    return datas, labels, imgs_path


import os
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix


def evaluate_model(model, loader, criterion, task_type, device, output_excel=None, output_feature_excel=None):
    model.eval()
    loss_sum = 0
    total = 0
    output_outputs = torch.Tensor([]).to(device)
    output_pred = torch.Tensor([]).to(device)
    labels = torch.IntTensor([]).to(device)

    # 创建一个列表来存储每个批次的数据和特征
    results_list = []
    feature_list = []
    names = []
    with torch.no_grad():
        for inputs, label, imgs_path in loader:
            inputs, label = inputs.float().to(device), label.unsqueeze(1).float().to(
                device) if task_type == 'binary' else label.to(device)
            outputs, feature = model(inputs)

            loss = criterion(outputs, label)
            total += 1
            loss_sum += loss.item()

            if task_type == 'binary':
                probabilities = torch.sigmoid(outputs)
                predicted = torch.round(probabilities)
                output_outputs = torch.cat((output_outputs, probabilities.float()), 0)
            else:
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                output_outputs = torch.cat((output_outputs, probabilities[:, 1].float()), 0)

            output_pred = torch.cat((output_pred, predicted.float()), 0)
            labels = torch.cat((labels, label.int()), 0)

            # 将当前批次的数据合并到结果列表中
            for i in range(len(imgs_path)):
                results_list.append({
                    'img': os.path.basename(imgs_path[i]),  # 图片路径
                    'label': label[i].item(),  # 真实标签
                    'predicted': predicted[i].item(),  # 预测标签
                    'probability': probabilities[i].cpu().numpy().tolist() if task_type == 'binary' else probabilities[
                        i].cpu().numpy().tolist(),  # 概率输出
                })
                # 将特征转换为列表并存储
                names.append(os.path.basename(imgs_path[i]))
                feature_list.append(feature[i].cpu().numpy().tolist())

    loss_sum = round(loss_sum / total, 4)
    if task_type == 'binary':
        conf_matrix = confusion_matrix(labels.cpu(), output_pred.cpu(), labels=[0, 1])
    else:
        conf_matrix = confusion_matrix(labels.cpu(), output_pred.cpu())

    results = get_result_binary(output_outputs.cpu(), output_pred.cpu(),
                                labels.cpu()) if task_type == 'binary' else None

    # 将结果列表转换为 DataFrame
    df_results = pd.DataFrame(results_list)

    # 保存分类结果为 .xlsx 文件
    if output_excel is not None:
        output_excel_path = os.path.dirname(output_excel)
        os.makedirs(output_excel_path, exist_ok=True)
        df_results.to_excel(output_excel, index=False)

    # 将特征列表转换为 DataFrame，并保存为 .xlsx 文件
    if output_feature_excel is not None:
        df = pd.DataFrame(feature_list)
        df.insert(0, 'Image', names)  # 将图像文件名添加为第一列
        df.to_excel(output_feature_excel, index=False)

    return loss_sum, conf_matrix, results


def load_model(model_name, num_classes, device, model_weight):
    if model_name == 'Resnet18':
        model = resnet18(num_classes=num_classes, sample_size=96, sample_duration=96).to(device)
    elif model_name == 'Resnet50':
        model = resnet50(num_classes=num_classes, sample_size=96, sample_duration=96).to(device)
    elif model_name == 'Resnet101':
        model = resnet101(num_classes=num_classes, sample_size=96, sample_duration=96).to(device)
    else:
        raise ValueError(f"Unknown model name {model_name}")

    model.load_state_dict(torch.load(model_weight, map_location=device), strict=False)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Model for Neoadjuvant Immunotherapy Dataset")
    parser.add_argument('--model_name', type=str, default='Resnet18', help='Model name')
    parser.add_argument('--model_weight', type=str, default='resnet18_results/fold_0/net_052.pth', help='Trained model weights path')
    parser.add_argument('--source_path', type=str, default='/mnt/e/skx/Breast-Ultrasound/Datasets/MRI-t1/cut/no_background_50/choose/4folds/fold_0/test', help='Test dataset path')
    parser.add_argument('--task_type', type=str, choices=['binary', 'multiclass'], default='binary', help='Task type')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes for classification')
    parser.add_argument('--output_excel', type=str, default='resnet18_results/fold_0/result.xlsx', help='Output Excel file path for results')
    parser.add_argument('--output_feature_excel', type=str, default='resnet18_results/fold_0/fold0_feature.xlsx', help='Output Excel file path for results')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model_name, args.num_classes, device, args.model_weight)
    test_imgs, test_labels, test_paths = load_and_preprocess(args.source_path, (96, 96, 96))
    test_dataset = Dataset_transform(imgs=test_imgs, labels=test_labels, paths=test_paths)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    criterion = nn.BCEWithLogitsLoss() if args.task_type == 'binary' else nn.CrossEntropyLoss()
    loss, conf_matrix, results = evaluate_model(model, test_loader, criterion, args.task_type, device, output_excel=args.output_excel, output_feature_excel=args.output_feature_excel)

    print("Test Loss:", loss)
    print("Confusion Matrix:\n", conf_matrix)
    if results:
        print("Metrics: Accuracy:", results[0], "| AUC:", results[1], "| Precision:", results[2],
              "| Recall:", results[3], "| F1 Score:", results[4])

