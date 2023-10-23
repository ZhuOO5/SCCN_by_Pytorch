import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
import numpy as np
from skimage.filters import threshold_otsu
from SCCN import SCCN, CCN
import cv2
import os
import numpy as np

architecture_config_opt ={
    'conv_config' : [
    (3, 3, 16, 1, 1), # (kernel_size, input_channels, nums_of_filter_as_out_put, stride, padding)
    ],

    'coupling_config' : [
        (1, 16, 20, 1, 0), 
        (1, 20, 20, 1, 0),
        (1, 20, 20, 1, 0),
    ]
}

architecture_config_sar ={
    'conv_config' : [
    (3, 1, 16, 1, 1), # (kernel_size, input_channels, nums_of_filter_as_out_put, stride, padding)
    ],

    'coupling_config' : [ 
        (1, 16, 20, 1, 0), 
        (1, 20, 20, 1, 0),
        (1, 20, 20, 1, 0),
    ]
}
def toImg(text, x, flag=False, type='opt'):
    # text: str, the name of the image
    # x: tensor, the image data
    # flag: bool, true for change map, false for dif map
    # type: str, 'opt' or 'sar', just for naming
    x = x.detach()
    x = x.cpu().numpy()
    maxValue = x.max()
    x = x*255/maxValue # to [0, 255]
    mat = np.uint8(x)
    if flag:
        # if flag is true, then the image is a change map
        # need to do thresholding
        threshold = threshold_otsu(mat)
        threshold, mat = cv2.threshold(mat, threshold, 255, cv2.THRESH_BINARY)
        print('threshold = ', threshold)
        text = 'change_map'+text
    else:
        text = 'dif_map'+text
    filename = f'./SCCN_SV1_OUTCOME/{text}.bmp'
    if not os.path.exists('./SCCN_SV1_OUTCOME'):
        os.mkdir('./SCCN_SV1_OUTCOME')
    cv2.imwrite(filename, mat)

def to_feature_img(text, x, imtype, layer):
    # text: str, part of the name of the image
    # x: tensor, the image data
    # imtype: str, 'opt' or 'sar', just for naming
    # layer: str, 'layer1', 'layer2', 'layer3' or 'layer4', just for naming
    x = x.detach()
    x = x.cpu().numpy()
    maxValue = x.max()
    x = x*255/maxValue # to [0, 255]
    mat = np.uint8(x)
    filename = f'./SCCN_SV1_{imtype}_{layer}/{text}.bmp'
    if not os.path.exists(f'SCCN_SV1_{imtype}_{layer}'):
        os.mkdir(f'./SCCN_SV1_{imtype}_{layer}')
    cv2.imwrite(filename, mat)

# Hyperparameters etc.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# with pretrained model, set 
LOAD_PRETRAINED_MODEL = True
EPOCHS = 1000

OPT_PRE_TRAINED_STATE_DICT= r'.\model\sv1_opt_new_model.pth'
SAR_PRE_TRAINED_STATE_DICT= r'.\model\sv1_sar_new_model.pth'
# OPT_PRE_TRAINED_STATE_DICT= r'.\model\yr1_opt_new_model.pth'
# SAR_PRE_TRAINED_STATE_DICT= r'.\model\yr1_sar_new_model.pth'
# OPT_PRE_TRAINED_STATE_DICT= r'.\model\fl1_opt_new_model.pth'
# SAR_PRE_TRAINED_STATE_DICT= r'.\model\fl1_sar_new_model.pth'


# sar_dir = r'Heterogeneous Data\Farmland\im1.bmp'
# opt_dir = r'Heterogeneous Data\Farmland\im2.bmp'
# sar_dir = r'Heterogeneous Data\Yellow River\im1.bmp'
# opt_dir = r'Heterogeneous Data\Yellow River\im2.bmp'
sar_dir = r'Heterogeneous Data\Shuguang Village\im1.bmp'
opt_dir = r'Heterogeneous Data\Shuguang Village\im2.bmp'


def revise_key_in_state_dict(orig_sate_dict):
    new_state_dict = {k.replace('old_prefix.', 'new_prefix.'): v for k, v in orig_sate_dict.items()}
    new_state_dict = {k.replace('ae1.forward_pass.0.', 'conv_layers.0.forward_pass.0.'): v for k, v in new_state_dict.items()}
    new_state_dict = {k.replace('coupling_daes.0.forward_pass.0.', 'coupling_layers.0.forward_pass.0.'): v for k, v in new_state_dict.items()}
    new_state_dict = {k.replace('coupling_daes.1.forward_pass.0.', 'coupling_layers.1.forward_pass.0.'): v for k, v in new_state_dict.items()}
    new_state_dict = {k.replace('coupling_daes.2.forward_pass.0.', 'coupling_layers.2.forward_pass.0.'): v for k, v in new_state_dict.items()}
    new_state_dict = {k: v for k, v in new_state_dict.items() if 'reconstruct' not in k}
    new_state_dict = {k:v for k,v in new_state_dict.items() if 'backard_pass' not in k}
    return new_state_dict

def main():    
    in_channels_opt = 3
    in_channels_sar = 1
    CCN_opt = CCN(in_channels_opt, architecture_config_opt).to(DEVICE)
    CCN_sar = CCN(in_channels_sar, architecture_config_sar).to(DEVICE)
    CCNs = {'CCN_opt':CCN_opt, 'CCN_sar':CCN_sar}

    if LOAD_PRETRAINED_MODEL:
            opt_state_dict = torch.load(OPT_PRE_TRAINED_STATE_DICT)
            opt_state_dict = revise_key_in_state_dict(opt_state_dict)
            CCN_opt.load_state_dict(opt_state_dict)
            sar_state_dict = torch.load(SAR_PRE_TRAINED_STATE_DICT)
            sar_state_dict = revise_key_in_state_dict(sar_state_dict)
            CCN_sar.load_state_dict(sar_state_dict)
    opt_img = cv2.imread(opt_dir)
    opt_img = transforms.ToTensor()(opt_img).cuda()
    sar_img = cv2.imread(sar_dir, 0) 
    sar_img = transforms.ToTensor()(sar_img).cuda()
    pro_shape = list(opt_img.shape[1:])
    pro_shape.append(1)
    pro_shape = tuple(pro_shape)
    pro_map = torch.rand(size=pro_shape).float().cuda()

    SCCN_T1 = SCCN(CCNs, pro_map)
    for epoch in range(EPOCHS+1):
        SCCN_T1.train()
        o1_opt, o2_opt, o3_opt, o4_opt, o1_sar, o2_sar, o3_sar, o4_sar, loss, dif_map = SCCN_T1(opt_img, sar_img)
        if epoch % 100 == 0:
            print("Saving epoch {}".format(epoch))   
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
            toImg(f'{epoch}', dif_map, flag=False)
            toImg(f'{epoch}', dif_map, flag=True)
            print("Loss for epoch {} is : {:.4f}".format(epoch, loss))
        
    print("Saving model...")
    torch.save(SCCN_T1.state_dict(), f'./model/SCCN_SV1_Epoch_{epoch}_loss_{loss}.pth')    


if __name__ == '__main__' :
    main() 