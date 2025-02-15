import os.path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# 读取Excel文件
file_paths = [
    r'E:\skx\Breast-Ultrasound\3D_CNN_pretrain\resnet18_results\result\MRI/fold0.xlsx',
    r'E:\skx\Breast-Ultrasound\3D_CNN_pretrain\resnet18_results\result\MRI/fold1.xlsx',
    r'E:\skx\Breast-Ultrasound\3D_CNN_pretrain\resnet18_results\result\MRI/fold2.xlsx',
    r'E:\skx\Breast-Ultrasound\3D_CNN_pretrain\resnet18_results\result\MRI/fold3.xlsx'
]

colors = ['green', 'purple', 'red', 'green']
pd_l1_models = ['fold0', 'fold1', 'fold2', 'fold3']

# 为每个文件绘制ROC曲线
for idx, file_path in enumerate(file_paths):
    data = pd.read_excel(file_path)

    # 提取标签和预测概率
    y_true = data['label'].values
    # y_prob = data['pred_score'].values
    # y_pred = data['pred_label'].values

    # for i in range(len(y_true)):
    #     if y_pred[i] == 0:
    #         y_prob[i] = 1 - y_prob[i]

    y_prob = data['probability'].apply(lambda x: float(x.strip("[]"))).values


    # 计算ROC曲线的fpr（1 - Specificity）、tpr（Sensitivity）和阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    specificity = 1 - fpr

    # 绘制Sensitivity-Specificity曲线
    plt.plot(specificity, tpr, color=colors[idx], lw=1.5, label=f'{pd_l1_models[idx]}  AUC={roc_auc:.3f}')

plt.xlabel('Specificity', fontsize=16)
plt.ylabel('Sensitivity', fontsize=16)
# plt.title('Sensitivity-Specificity Curve', fontsize=16)
plt.legend(loc='lower left', fontsize=18)
plt.tight_layout()

plt.savefig(r'E:\skx\Breast-Ultrasound\3D_CNN_pretrain\resnet18_results\result\MRI\Ultrasound_result.png', dpi=300, bbox_inches='tight')  # Save with 300 dpi for higher quality

plt.show()





