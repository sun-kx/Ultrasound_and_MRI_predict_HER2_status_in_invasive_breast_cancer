import numpy as np
import matplotlib.pyplot as plt
# from monai.visualize import GradCAM
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
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from PIL import Image
import torch.nn.functional as F
import numpy as np

# 定义 Grad-CAM 类
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, label, criterion, target_size=None):
        """
        计算 Grad-CAM 并返回插值到目标尺寸的热图。
        :param input_tensor: 模型输入，形状为 [B, C, D, H, W]。
        :param label: 目标类别标签，形状为 [B, num_classes]。
        :param criterion: 损失函数，用于计算目标类别的梯度。
        :param target_size: 热图的目标尺寸 (D, H, W)，如果为 None 则返回原始尺寸。
        :return: 插值后的热图，形状为 target_size。
        """
        # 前向传播
        output = self.model(input_tensor)
        loss = criterion(output, label)

        # 反向传播
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        # 提取梯度和特征
        gradients = self.gradients
        activations = self.activations
        # 计算权重
        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)  # 平均池化梯度

        # 使用权重加权特征
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(input_tensor.device)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i]

        # 应用 ReLU
        cam = F.relu(cam)

        # 插值到目标尺寸
        if target_size is not None:
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=target_size, mode='trilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()
        else:
            cam = cam.cpu().numpy()

        return cam

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


def generate_gradcam_visualizations(model, loader, criterion, device, output_dir, layer_name):
    """
    使用 Grad-CAM 生成热图并保存
    :param model: 训练好的模型
    :param loader: DataLoader 包含测试数据
    :param device: 设备 (CPU/GPU)
    :param output_dir: 保存热图的目录
    :param layer_name: 模型中用于 Grad-CAM 的目标层
    """
    model.eval()
    for inputs, labels, paths in loader:
        inputs = inputs.float().to(device)
        print(inputs.shape)
        inputs.requires_grad = True
        labels = labels.unsqueeze(1).float().to(device)
        target_layer = model.layer4
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam(inputs, labels, criterion,target_size=(96,96,96))
        img = inputs[0][0]

        overlay = np.clip(img + cam, 0, 1)
        out_file = os.path.join(output_dir, os.path.basename(paths[0]))

        for i in range(overlay.shape[0]):
            out_file_cam = os.path.join(out_file, 'grad_cam')
            os.makedirs(out_file_cam, exist_ok=True)
            plt.imshow(overlay[i,:,:], cmap="jet")
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(out_file_cam, f"{os.path.basename(paths[0])}_{i}.png"), bbox_inches="tight")
            plt.close()

            out_file_img = os.path.join(out_file, 'img')
            os.makedirs(out_file_img, exist_ok=True)
            plt.imshow(img[i,:,:].cpu().detach().numpy(), cmap="gray")
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(out_file_img, f"{os.path.basename(paths[0])}_{i}.png"), bbox_inches="tight")
            plt.close()







        # targets = None

        #
        # with GradCAM(model=model, target_layers=[model.layer4],
        #              use_cuda=torch.cuda.is_available()) as cam:
        #
        #     grayscale_cam = cam(input_tensor=inputs, targets=targets).astype(np.float32)[0, :]


            # print(grayscale_cam.shape)
            # # print(val_in[0, :, 12,:,:].shape)
            # # print(grayscale_cam)
            # # print(grayscale_cam.shape,grayscale_cam.max(),grayscale_cam.min(),grayscale_cam.mean(),idx)
            # grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
            # # 逐张显示每个切片
            # for i in range(grayscale_cam.shape[2]):  # 遍历第三个维度 (Z 轴)
            #         img = Image.fromarray(255*grayscale_cam[:, :, i].astype(np.uint8))
            #         img.save("{}.png".format(i))

            # print(grayscale_cam)
            # val_inp = np.float32(inputs[0, 0, :, :, :, None].cpu())
            # val_inp /= val_inp.max()

            # cam_image = show_cam_on_image(val_inp, grayscale_cam, use_rgb=True)
            # print(cam_image.shape)
            # img = Image.fromarray(cam_image)
            # # img.show()
            # img.save("/cache/SZG/SOTA_SZGmodelV1/models/cam_unet2/{}_0_cnnup.png".format(idx))


        #
        # # 获取 Grad-CAM 热图
        # heatmaps = grad_cam(x=inputs)
        #
        #
        # # 遍历当前批次的所有样本
        # for i in range(len(paths)):
        #
        #     # img = inputs[i].cpu().numpy()[0]  # 单通道 3D 图像
        #     img = inputs[i].cpu()[0]  # 单通道 3D 图像
        #     heatmap = heatmaps[i].cpu().numpy()[0]
        #
        #     # 对热图进行归一化
        #     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        #
        #     # 保存热图
        #     output_file = os.path.join(output_dir, f"{os.path.basename(paths[i])}_gradcam.nii.gz")
        #     save_heatmap_as_image(img, heatmap, output_file)

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
    parser.add_argument('--model_weight', type=str, default='resnet18_results/fold_3/net_028.pth',
                        help='Trained model weights path')
    parser.add_argument('--source_path', type=str,
                        default='/mnt/e/skx/Breast-Ultrasound/Datasets/MRI-t1/cut/no_background_50/choose/4folds/fold_3/test',
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
    criterion = nn.BCEWithLogitsLoss()
    generate_gradcam_visualizations(model, test_loader, criterion, device, args.output_dir, args.layer_name)
    print(f"Grad-CAM visualizations saved in: {args.output_dir}")
