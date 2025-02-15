import numpy as np
import matplotlib.pyplot as plt
from monai.visualize import GradCAM
from monai.transforms import SaveImage
import os
import torch
import pandas as pd
import argparse
import torch.nn as nn
import SimpleITK as sitk
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
from PIL import Image
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

def generate_gradcam_visualizations(model, loader, device, output_dir, layer_name):
    """
    使用 Grad-CAM 生成热图并保存
    :param model: 训练好的模型
    :param loader: DataLoader 包含测试数据
    :param device: 设备 (CPU/GPU)
    :param output_dir: 保存热图的目录
    :param layer_name: 模型中用于 Grad-CAM 的目标层
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # 使用 MONAI 的 Grad-CAM 方法
    grad_cam = GradCAM(
        nn_module=model,
        target_layers=[layer_name],  # 指定用于 Grad-CAM 的目标层
        # use_cuda=torch.cuda.is_available()
    )


    for inputs, labels, paths in loader:
        inputs = inputs.float().to(device)
        inputs.requires_grad = True
        labels = labels.to(device)

        # 获取 Grad-CAM 热图
        heatmaps = grad_cam(x=inputs)


        # 遍历当前批次的所有样本
        for i in range(len(paths)):

            # img = inputs[i].cpu().numpy()[0]  # 单通道 3D 图像
            img = inputs[i].cpu()[0]  # 单通道 3D 图像
            heatmap = heatmaps[i].cpu().numpy()[0]

            # 对热图进行归一化
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            for i in range(heatmap.shape[2]):  # 遍历第三个维度 (Z 轴)
                # 将当前切片数据转换为 uint8 类型
                slice_data = (255 * heatmap[:, :, i]).astype(np.uint8)

                # 使用 PIL.Image.fromarray 创建图像
                img = Image.fromarray(slice_data)

                # 保存为 PNG 文件
                img.save("slice_{}.png".format(i))
            # 保存热图
            output_file = os.path.join(output_dir, f"{os.path.basename(paths[i])}_gradcam.nii.gz")
            save_heatmap_as_image(img, heatmap, output_file)

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

def save_heatmap_as_image(image, heatmap, output_file):
    """
    保存 Grad-CAM 热图叠加到原始图像的可视化结果
    """
    overlay = np.clip(image + heatmap, 0, 1)  # 将原图和热图叠加
    sitk_img = sitk.GetImageFromArray(overlay)
    sitk.WriteImage(sitk_img, output_file)

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
    parser = argparse.ArgumentParser(
        description="Test Model for Neoadjuvant Immunotherapy Dataset with Grad-CAM Visualization")
    parser.add_argument('--model_name', type=str, default='Resnet18', help='Model name')
    parser.add_argument('--model_weight', type=str, default='resnet18_results/fold_0/net_052.pth',
                        help='Trained model weights path')
    parser.add_argument('--source_path', type=str,
                        default='/mnt/e/skx/Breast-Ultrasound/Datasets/MRI-t1/cut/no_background_50/choose/4folds/fold_0/test',
                        help='Test dataset path')
    parser.add_argument('--task_type', type=str, choices=['binary', 'multiclass'], default='binary', help='Task type')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes for classification')
    parser.add_argument('--output_dir', type=str, default='gradcam_results',
                        help='Directory to save Grad-CAM visualizations')
    parser.add_argument('--layer_name', type=str, default='layer4', help='Target layer name for Grad-CAM')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model_name, args.num_classes, device, args.model_weight)
    test_imgs, test_labels, test_paths = load_and_preprocess(args.source_path, (96, 96, 96))
    test_dataset = Dataset_transform(imgs=test_imgs, labels=test_labels, paths=test_paths)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 每次处理一张图片以生成 Grad-CAM 热图

    generate_gradcam_visualizations(model, test_loader, device, args.output_dir, args.layer_name)
    print(f"Grad-CAM visualizations saved in: {args.output_dir}")
