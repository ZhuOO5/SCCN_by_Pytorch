# pre_train.py
# pre-training for SCCN by using DAEs
from DAE import DAEs, DAE
import os
import time 
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# to save the reconstructed image
def toImg(text, x, imtype):
    x = x.detach()
    x = x.cpu().numpy()
    maxValue = x.max()
    x = x*255/maxValue # value to [0, 255]
    mat = np.uint8(x)
    mat = mat.transpose(1, 2, 0)
    filename = f'./sv1_features_recon1/{text}.bmp'
    if not os.path.exists('./sv1_features_recon1'):
        os.mkdir('./sv1_features_recon1')
    cv2.imwrite(filename, mat)

# to save the feature map
def to_feature_img(text, x, imtype, layer):
    x = x.detach()
    x = x.cpu().numpy()
    maxValue = x.max()
    x = x*255/maxValue # value to [0, 255]
    mat = np.uint8(x)
    filename = f'./sv1_features_{imtype}_{layer}/{text}.bmp'
    if not os.path.exists(f'sv1_features_{imtype}_{layer}'):
        os.mkdir(f'./sv1_features_{imtype}_{layer}')
    cv2.imwrite(filename, mat)

# the difference between the structure of DAEs for opt and sar is the input channels
architecture_config_opt ={
    'conv_config' : [
    (3, 3, 32, 1, 1), # (kernel_size, input_channels, nums_of_filter_as_out_put, stride, padding)
    ],

    'coupling_config' : [ 
        (1, 32, 20, 1, 0), 
        (1, 20, 20, 1, 0),
        (1, 20, 20, 1, 0),
    ]
}
architecture_config_sar ={
    'conv_config' : [
    (3, 1, 32, 1, 1), # (kernel_size, input_channels, nums_of_filter_as_out_put, stride, padding)
    ],

    'coupling_config' : [ 
        (1, 32, 20, 1, 0), 
        (1, 20, 20, 1, 0),
        (1, 20, 20, 1, 0),
    ]
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 1000

# sar_dir = r'Heterogeneous Data\Farmland\im1.bmp'
# opt_dir = r'Heterogeneous Data\Farmland\im2.bmp'
# sar_dir = r'Heterogeneous Data\Yellow River\im1.bmp'
# opt_dir = r'Heterogeneous Data\Yellow River\im2.bmp'
sar_dir = r'Heterogeneous Data\Shuguang Village\im1.bmp'
opt_dir = r'Heterogeneous Data\Shuguang Village\im2.bmp'

opt_img = cv2.imread(opt_dir)
opt_img = transforms.ToTensor()(opt_img).cuda()
opt_img_info = {'img_type':'opt'}

sar_img = cv2.imread(sar_dir, 0) # read as gray image
sar_img = transforms.ToTensor()(sar_img).cuda()
sar_img_info = {'img_type':'sar', 'looks_of_sensor':1}

# save original potical and sar image
toImg('orig_sar', sar_img, 'sar')
toImg('orig_opt', opt_img, 'opt')

model_sar = DAEs(sar_img_info, architecture_config_sar).to(DEVICE)
model_opt = DAEs(opt_img_info, architecture_config_opt).to(DEVICE)

for epoch in range(NUM_EPOCHS+1):
    model_sar.train()
    model_opt.train()
    total_time = time.time()
    o1, o2, o3, o4 = model_opt(opt_img) 
    o1, o2, o3, o4 = model_sar(sar_img)
    total_time = time.time() - total_time

    model_opt.eval()
    model_sar.eval()
    o1_opt, o2_opt, o3_opt, o4_opt, opt_reconstructed = model_opt(opt_img)
    o1_sar, o2_sar, o3_sar, o4_sar, sar_reconstructed = model_sar(sar_img)
    reconstruction_loss_fn = nn.MSELoss()
    opt_loss = reconstruction_loss_fn(opt_img, opt_reconstructed)
    sar_loss = reconstruction_loss_fn(sar_img, sar_reconstructed)
    
    if epoch % 100 == 0:
        # save the feature map of each layer
        # print("Saving epoch {}".format(epoch))   
        # o1_opt = o1_opt.permute(1, 2, 0)
        # o1_sar = o1_sar.permute(1, 2, 0)
        # o2_opt = o2_opt.permute(1, 2, 0)
        # o2_sar = o2_sar.permute(1, 2, 0)
        # o3_opt = o3_opt.permute(1, 2, 0)
        # o3_sar = o3_sar.permute(1, 2, 0)
        # o4_opt = o4_opt.permute(1, 2, 0)
        # o4_sar = o4_sar.permute(1, 2, 0)
        # n1 = o1_opt.shape[-1]
        # for i in range(n1):
        #     to_feature_img(f'o1_feature_{i}_{epoch}',o1_opt[:, :, i],'opt', layer='layer1')
        #     to_feature_img(f'o1_feature_{i}_{epoch}',o1_sar[:, :, i],'sar', layer='layer1')
        # n2 = o2_opt.shape[-1]
        # for i in range(n2):
        #     to_feature_img(f'o2_feature_{i}_{epoch}',o2_opt[:, :, i],'opt', layer='layer2')
        #     to_feature_img(f'o2_feature_{i}_{epoch}',o2_sar[:, :, i],'sar', layer='layer2')
        #     to_feature_img(f'o3_feature_{i}_{epoch}',o3_opt[:, :, i],'opt', layer='layer3')
        #     to_feature_img(f'o3_feature_{i}_{epoch}',o3_sar[:, :, i],'sar', layer='layer3')
        #     to_feature_img(f'o4_feature_{i}_{epoch}',o4_opt[:, :, i],'opt', layer='layer4')
        #     to_feature_img(f'o4_feature_{i}_{epoch}',o4_sar[:, :, i],'sar', layer='layer4')
        # save the reconstructed image
        toImg(f'opt_reconstructed_{epoch}', opt_reconstructed, 'opt')
        toImg(f'sar_reconstructed_{epoch}', sar_reconstructed, 'sar')
    print("Epoch {} complete\tTime: {:.4f}s\t\toptLoss: {:.4f}\t\tsar_Loss: {:.4f}".format(epoch, total_time, opt_loss, sar_loss))

# save the model
torch.save(model_opt.state_dict(), './model/sv1_opt_new_model.pth')  
torch.save(model_sar.state_dict(), './model/sv1_sar_new_model.pth')

