import csv
import pickle

# 打开.pkl文件并加载数据
with open(r'E:\skx\Breast-Ultrasound\mmpretrain\work_dirs\convnext_v2_MRI_50_1\fold_0/out.pkl', 'rb') as f:
    data = pickle.load(f)

# 创建CSV文件并写入数据
with open(r'E:\skx\Breast-Ultrasound\mmpretrain\work_dirs\convnext_v2_MRI_50_1\fold_0/fold0.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 写入表头
    writer.writerow(['sample_idx', 'img_path', 'gt_label', 'pred_label', 'pred_score', 'num_classes', 'ori_shape', 'img_shape'])

    # 写入每一行的数据
    for item in data:
        pred_score = 0
        ori_shape = item['ori_shape']
        img_shape = item['img_shape']
        # scale_factor = item['scale_factor']
        sample_idx = item['sample_idx']
        img_path = item['img_path']
        num_classes = item['num_classes']
        # pred_score = ','.join(str(score.item()) for score in item['pred_score'])
        for score in item['pred_score']:
            pred_score = max(score.item(), pred_score)
        gt_label = item['gt_label'].item()
        pred_label = item['pred_label'].item()

        writer.writerow([sample_idx, img_path, gt_label, pred_label, pred_score, num_classes, ori_shape, img_shape])

print("转换完成！")

