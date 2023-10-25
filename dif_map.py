import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage.filters import threshold_otsu
# for adjusting the outcome difference map of yellow river
dm_dir = r".\SCCN_YR1_OUTCOME\dif_map1000.bmp"
brighter_map = r".\SCCN_YR1_OUTCOME\brighter_dif_map1000.bmp"
# dm = cv2.imread(dm_dir, cv2.IMREAD_GRAYSCALE)
dm = Image.open(dm_dir).convert('L')
dm = F.adjust_brightness(dm, brightness_factor=8) # 亮度增强
dm_tensor = F.pil_to_tensor(dm)
# # threshold = threshold_otsu(dm)
# th, change_map = cv2.threshold(dm, threshold, 255, cv2.THRESH_BINARY)
# change_map.save(r".\SCCN_YR3_OUTCOME\brignter_change_map1000.bmp")
# dm.save(brighter_map)
# hist = dm.histogram()
# dm_tensor = torch.from_numpy(dm)
img = F.to_pil_image(dm_tensor)
img.save(brighter_map)
new_dm = cv2.imread(brighter_map, cv2.IMREAD_GRAYSCALE)
threshold = threshold_otsu(new_dm)
th, change_map = cv2.threshold(new_dm, threshold, 255, cv2.THRESH_BINARY)
change_map = F.to_pil_image(change_map)
change_map.save(r".\SCCN_YR1_OUTCOME\brignter_change_map1000.bmp")
dm_tensor = dm_tensor.type(torch.float32)

hist = torch.histc(dm_tensor, bins=256, min=0, max=255)
# torch.hist 该函数将返回一个长度为256的张量，其中每个元素表示差异图中对应像素值的出现次数。您可以使用这些信息来了解差异图像素值的分布情况。12
# Assuming your histogram data is stored in a PyTorch tensor called `hist`
plt.bar(range(len(hist)), hist)
plt.title('Histogram of Difference Map Pixel Values')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

