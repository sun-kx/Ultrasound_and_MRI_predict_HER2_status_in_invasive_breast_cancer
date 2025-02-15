import os
import pandas as pd
from mmpretrain import FeatureExtractor


config = 'convnextv2_configs.py'
checkpoint = 'work_dirs/convnext_v2_MRI_50_1/fold_2/best_single-label_f1-score_epoch_77.pth'
# 初始化特征提取器
inferencer = FeatureExtractor(model=config, pretrained=checkpoint, device='cuda')


def extract_and_save_features(folder_path, output_excel):
    features_list = []
    image_paths = []

    # 遍历文件夹中的所有图像
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):  # 只处理常见图像格式
            image_path = os.path.join(folder_path, filename)

            # 提取特征
            feats = inferencer(image_path)[0][0]

            # 将图像文件名和特征添加到列表中
            image_paths.append(filename)
            # 将特征转换为一维数组，并将其添加到 features_list
            features_list.append(feats.cpu().numpy().tolist())

    # 创建 DataFrame
    df = pd.DataFrame(features_list)
    df.insert(0, 'Image', image_paths)  # 将图像文件名添加为第一列

    # 保存为 Excel 文件
    df.to_excel(output_excel, index=False)
    print(f"特征已保存至 {output_excel}")


# 示例使用
folder_path = r"/mnt/e/skx/Breast-Ultrasound/Datasets/Ultrasound/choose/MRI-t1/4folds/fold_2/org"
output_excel = r"work_dirs/convnext_v2_MRI_50_1/fold_2/fold2_features.xlsx"
extract_and_save_features(folder_path, output_excel)
