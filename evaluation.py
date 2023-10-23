# 对比最后的difference map和refferrence image
# 以像素为单位去进行对比
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage.filters import threshold_otsu
# from sklearn.metrics import roc_auc_score

def evaluation_for_difference_map(dm, rm, th):
    #dm : difference map, SCCN通过两张图片生成的difference map, h * w（0-255）
    #rm : reference map, 人工标记, h * w 
    # 返回根据不同门限值得到的标记
    tpr = [0, 1]
    fpr = [0, 1]
    orig_dm = dm
    orig_rm = rm
    for i in range(1, 256):
        threshold = i
        rm = orig_rm
        dm = orig_dm
        ret, rm = cv2.threshold(rm, threshold, 1, cv2.THRESH_BINARY)
        ret, dm = cv2.threshold(dm, threshold, 1, cv2.THRESH_BINARY)
        zero = np.zeros(rm.shape)
        one = np.ones(rm.shape)
        TP = np.where(dm + rm ==2, one, zero).sum()
        TN = np.where(dm + rm ==0, one, zero).sum()
        FP = np.where((dm==1) & (rm==0), one, zero).sum()
        FN = np.where((dm==0) & (rm==1), one, zero).sum()
        OE = FN + FP
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        if FPR != 0:
            tpr.append(TPR)
            fpr.append(FPR)
        # if TPR and FPR:
        #     tpr.append(TPR)
        #     fpr.append(FPR)


        if i==th:
            CA = (TP+TN)/(TP+TN+FP+FN)
            PRE = (TP+FP)*(TP+FN)/np.square(TP+TN+FP+FN)+\
            (FN+TN)*(FP+TN)/np.square(TP+TN+FP+FN)
            KC = (CA-PRE)/(1.0-PRE)
            print('FP', FP)
            print('FN', FN)
            print('TPR:',TPR)
            print('FPR:',FPR)
            print('PRE:',PRE)
            print('OE', OE)
            print('CA:',CA)
            print('KC', KC)
        # coordinate.append((FPR, TPR))
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    fpr, indices = np.unique(fpr, return_index=True)
    tpr = tpr[indices]
    sort_index = np.argsort(fpr)
    fpr_sorted = fpr[sort_index]
    tpr_sorted = tpr[sort_index]
    return tpr_sorted, fpr_sorted



# rm_dir = r'.\Heterogeneous Data\Farmland\reference.bmp'
rm_dir = r'Heterogeneous Data\Shuguang Village\reference.bmp'
# rm_dir = r'Heterogeneous Data\Huanghe River\Yellow River\im3.bmp'
# rm_dir = r'Heterogeneous Data\Yellow River\reference.bmp'
rm = cv2.imread(rm_dir, cv2.IMREAD_GRAYSCALE)
# dm = np.random.randint(0, 256, size=rm.shape, dtype=np.uint8) # 随机生成的
# dm_dir = r'.\imgs37\dif_map_180.bmp'
# dm_dir =r'imgs64\dif_map_400.bmp'
# dm_dir = r".\SCCN_YR3_OUTCOME\brignter_dif_map1000.bmp"
dm_dir = r".\SCCN_SV1_OUTCOME\dif_map1000.bmp"
# dm_dir = r'sv_new_outcome1\dif_map450.bmp'
dm = cv2.imread(dm_dir, cv2.IMREAD_GRAYSCALE)
threshold = threshold_otsu(dm)
print('threshold = ', threshold)
tpr, fpr = evaluation_for_difference_map(dm, rm, threshold)
# fpr = [i for i in fpr if not np.isnan(i)]
# tpr = [i for i in tpr if not np.isnan(i)]
# 假设fpr和tpr是你的数据
fpr = np.array(fpr)
tpr = np.array(tpr)

# 检查并处理NaN值
if np.isnan(fpr).any() or np.isnan(tpr).any():
    print("Warning: NaN values found and replaced by zeros.")
    fpr = np.nan_to_num(fpr)
    tpr = np.nan_to_num(tpr)
# auc_score = roc_auc_score(fpr, tpr)
auc = auc(fpr, tpr)
# print('auc:',)
#画图
plt.figure()
lw = 2

plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='test_curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title(f'ROC Curve,AUC:{auc_score}')
plt.title(f'ROC Curve,AUC:{auc}')
plt.legend(loc="lower right")
plt.show()




# def ROC():
    

