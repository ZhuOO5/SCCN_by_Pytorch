# SCCN
import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, layer_config):
        super(CNNBlock, self).__init__()
        self.layer_config = layer_config 
        self.kernel_size = layer_config[0]
        self.input_channels = layer_config[1]
        self.output_channels = layer_config[2]
        self.stride = layer_config[3]
        self.padding = layer_config[4]
        self.forward_pass = nn.Sequential(
            nn.Conv2d(self.input_channels, self.output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.forward_pass(x)
# actually coupling is a 1 x 1 convolution
# so the CouplingBlock is just the same as CNNBlock, you can delete it, but I keep it for the sake of clarity
class CouplingBlock(nn.Module):
    def __init__(self, layer_config) : 
        super(CouplingBlock, self).__init__()
        self.layer_config = layer_config 
        self.kernel_size = layer_config[0]
        self.input_channels = layer_config[1]
        self.output_channels = layer_config[2]
        self.stride = layer_config[3]
        self.padding = layer_config[4]
        self.forward_pass = nn.Sequential(
            nn.Conv2d(self.input_channels, self.output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.Sigmoid()
        )

    def forward(self, x): 
        return self.forward_pass(x)

class CCN(nn.Module):
    # CCN: Convolutional Coupling Network
    # CCN is a network composed of several CNNBlocks and CouplingBlocks
    # its half of the structure of SCCN
    def __init__(self, in_channels,  architecture_config, **kwargs): 
        super(CCN, self).__init__()
        self.in_channels = in_channels
        self.conv_config = architecture_config['conv_config']
        self.coupling_config =architecture_config['coupling_config']
        self.conv_layers = nn.Sequential(*[CNNBlock(t) for t in self.conv_config])
        self.coupling_layers = nn.Sequential(*[CouplingBlock(t) for t in self.coupling_config])
        self.outputs = []


    def forward(self, x):
        a1 = self.conv_layers[0](x)
        a2 = self.coupling_layers[0](a1)
        a3 = self.coupling_layers[1](a2)
        a4 = self.coupling_layers[2](a3)
        return a1, a2, a3, a4  

class SCCN_Loss(nn.Module):
    def __init__(self, lam=0.15):
        super(SCCN_Loss, self).__init__()
        self.lam = lam
    
    def forward(self, dif_map, unch_pro):
        # coupling dunction, the training process is to minimize this function
        loss = torch.sum(dif_map * unch_pro)-self.lam * torch.sum(unch_pro)
        return loss


class SCCN(nn.Module):
    def __init__(self, CCNs:dict, unch_pro) -> None:
        super(SCCN, self).__init__()
        self.CCN_opt = CCNs['CCN_opt']
        self.CCN_sar = CCNs['CCN_sar']
        # self.optimizer = torch.optim.Adam(self.CCN_opt.parameters(), lr=4e-4) # fix the sar part, only train the opt part(this is the original setting)
        self.optimizer = torch.optim.Adam(self.CCN_sar.parameters(), lr=4e-4) # fix the opt part, only train the sar part
        self.loss = SCCN_Loss(lam=0.15)
        self.unch_pro = unch_pro # to record the unchange probability for each pixel

    def update_unch_pro(self, opt_img, sar_img):
        self.eval()
        o1_opt, o2_opt, o3_opt, o4_opt = self.CCN_opt(opt_img)
        o1_sar, o2_sar, o3_sar, o4_sar = self.CCN_sar(sar_img)
        # differnce map
        d_map = torch.sqrt(
            torch.sum(
                torch.square(torch.sub(o4_opt, o4_sar)), dim=0, keepdim=True
            )
        ).permute(1, 2, 0)
        # normalize the difference map
        d_map = (d_map - d_map.min()) / (d_map.max() - d_map.min())
        zero = torch.zeros_like(d_map)
        one = torch.ones_like(d_map)
        # update the unchange probability map
        unch_pro = torch.where(d_map < self.loss.lam, one, zero)
        self.unch_pro = unch_pro

    def forward(self, opt_img, sar_img):
        self.train()
        o1_opt, o2_opt, o3_opt, o4_opt = self.CCN_opt(opt_img)
        o1_sar, o2_sar, o3_sar, o4_sar = self.CCN_sar(sar_img)
        # d_map = torch.sqrt(
        #     torch.sum(
        #         torch.square(o4_opt-o4_sar), dim=0, keepdim=True
        #     )
        # ).permute(1, 2, 0)
        # calculate the difference map
        d_map = torch.sqrt(
            torch.sum(
                torch.square(torch.sub(o4_opt, o4_sar)), dim=0, keepdim=True
            )
        ).permute(1, 2, 0)
        # normalize the difference map to [0,1]
        d_map = (d_map - d_map.min()) / (d_map.max() - d_map.min())
        
        if self.training:
            loss = self.loss(d_map, self.unch_pro)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_unch_pro(opt_img,sar_img)
            # I return the output of each layer for the sake of visualization, 
            # you can change the return to a more concise form as you like
            return o1_opt, o2_opt, o3_opt, o4_opt, o1_sar, o2_sar, o3_sar, o4_sar, loss, d_map
        return o1_opt, o2_opt, o3_opt, o4_opt, o1_sar, o2_sar, o3_sar, o4_sar, d_map
    
    
