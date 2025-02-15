import pickle
import os
import shutil
# 读取 .pkl 文件
with open('no_background_50_best_level7/out.pkl', 'rb') as f:  # 'rb' 表示以二进制方式读取文件
    datas1 = pickle.load(f)

#     print(datas[0]['pred_label'].tolist()[0])
#
for data in datas1:
    pred_score = data['pred_score'].tolist()
    data_path = data['img_path'].replace('/mnt/e', 'E:')
    if data['pred_label'] == data['gt_label']:
        if (data['pred_label'].tolist()[0]==0 and max(pred_score) >= 0.8) or (data['pred_label'].tolist()[0]==1 and max(pred_score) >= 0.7):
            print(data_path)
            new_img = data_path.replace('best_level7/All', 'best_level7/best_add')
            # 如果输出文件夹不存在，创建它
            new_img_path = os.path.dirname(new_img)
            if not os.path.exists(new_img_path):
                os.makedirs(new_img_path)
            print(data['img_path'], new_img)
            shutil.copy(data_path, new_img)

# # 读取 .pkl 文件
# with open('no_background_50_best_level3_0/out.pkl', 'rb') as f:  # 'rb' 表示以二进制方式读取文件
#     datas1 = pickle.load(f)
# with open('no_background_50_best_level3_1/out.pkl', 'rb') as f:  # 'rb' 表示以二进制方式读取文件
#     datas2 = pickle.load(f)
# with open('no_background_50_best_level3_2/out.pkl', 'rb') as f:  # 'rb' 表示以二进制方式读取文件
#     datas3 = pickle.load(f)
#
# #     print(datas[0]['pred_label'].tolist()[0])
# #
# for data1 in datas1:
#     for data2 in datas2:
#         for data3 in datas3:
#             if data3['img_path']==data1['img_path'] and data2['img_path']==data1['img_path']:
#                 # pred_score = data['pred_score'].tolist()
#                 data_path = data1['img_path'].replace('/mnt/e', 'E:')
#                 if data1['pred_label'] == data1['gt_label'] and data2['pred_label'] == data2['gt_label'] and data3['pred_label'] == data3['gt_label']:
#                     # if (data['pred_label'].tolist()[0]==0 and max(pred_score) > 0.70) or (data['pred_label'].tolist()[0]==1 and max(pred_score) > 0.60):
#                     new_img = data_path.replace('best_level3', 'best_level3/best')
#                     # 如果输出文件夹不存在，创建它
#                     new_img_path = os.path.dirname(new_img)
#                     if not os.path.exists(new_img_path):
#                         os.makedirs(new_img_path)
#                     print(data1['img_path'], new_img)
#                     shutil.copy(data_path, new_img)


# 打印读取的数据
#
# # print(data)
#
# a = os.listdir(r'E:\skx\Breast-Ultrasound\Datasets\choose\best1\0')
# b = os.listdir(r'E:\skx\Breast-Ultrasound\Datasets\choose\best1\1')
# c = os.listdir(r'E:\skx\Breast-Ultrasound\Datasets\choose\best2\0')
# d = os.listdir(r'E:\skx\Breast-Ultrasound\Datasets\choose\best2\1')
#
# e = set(c) & set(a)
# f = set(d) & set(b)
#
# for i in e:
#     e_path = os.path.join(r'E:\skx\Breast-Ultrasound\Datasets\cut\no_background_50\All\0', i)
#     out_path_0 = os.path.join(r'E:\skx\Breast-Ultrasound\Datasets\choose\best\0', i)
#     shutil.copy(e_path, out_path_0)
# for j in f:
#     d_path = os.path.join(r'E:\skx\Breast-Ultrasound\Datasets\cut\no_background_50\All\1', j)
#     out_path_1 = os.path.join(r'E:\skx\Breast-Ultrasound\Datasets\choose\best\1', j)
#     shutil.copy(d_path, out_path_1)



