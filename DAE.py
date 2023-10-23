# Denoising AutoEncoder, to initialize the weights of SCCN
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DAE(nn.Module): # a DAE for a layer
    def __init__(self, layer_config, input_info):
        # layer_config (turple): the structure of current layer
        # (kernel_size, input_channels, nums_of_filter_as_out_put, stride, padding)
        # input_info (dictionary) : record some information about input, such as 'input_type' for noise adding
        #                           looks_of_sensor for GammaNoise
        super(DAE, self).__init__()
        self.input_info = input_info
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
        # the process of decoding
        self.backard_pass = nn.Sequential(
            nn.ConvTranspose2d(self.output_channels, self.input_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.Sigmoid()

        )
        self.to(DEVICE)
        self.loss_fn = nn.MSELoss()
        # For pre-training, we can choose different learning rate for different image type
        if self.input_info['img_type'] == 'opt':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)   
        elif self.input_info['img_type'] == 'sar':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=4e-2)
    
    def forward(self, x):
        # There is no noise adding because we only add noise at the input of DAEs
        x = x.detach()
        y = self.forward_pass(x)

        if self.training:
            x_reconstuct = self.reconstruct(y)
            loss = self.loss_fn(x_reconstuct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
        
        return y.detach() # return the output of current layer
    
    def reconstruct(self, x):
        return self.backard_pass(x)


class DAEs(nn.Module):
    def __init__(self,img_info, architecture_config): 
        super(DAEs, self).__init__()
        self.conv_config = architecture_config['conv_config']
        self.coupling_config =architecture_config['coupling_config']
        self.img_info = img_info # A dictionary, record the type of image, and the looks of sensor for SAR image, in my experiments, looks_of_sensor = 1
        print('Create DAEs')
        
        self.ae1 = DAE(self.conv_config[0],img_info)
        self.coupling_daes = nn.ModuleList([DAE(t, img_info) for t in self.coupling_config])

    def add_GaussianNoise(self, input, std = 0.1, mean = 0):
        # add noise for optical image
        noise = torch.randn(input.shape) * std + mean 
        noise = noise.cuda()
        input_with_noise = input + noise
        return input_with_noise
    
    def add_GammaNoise(self,input,looks_of_sensor):
        # add noise for SAR image
        concentration = looks_of_sensor
        rate = 1 / looks_of_sensor
        gamma_dist = torch.distributions.Gamma(concentration, rate)
        noise = gamma_dist.sample(input.shape)
        noise = torch.nn.functional.normalize(noise, p=2.0, dim=0)
        noise = noise.cuda()
        input_with_noise = input * noise
        return input_with_noise  
     
    def forward(self, x):
        if self.training:
            self.ae1.training = True
            for ae in self.coupling_daes:
                ae.training = True
        else:
            self.ae1.training = False
            for ae in self.coupling_daes:
                ae.training = False
        # Adding noise
        if self.img_info['img_type'] == 'opt':
            x = self.add_GaussianNoise(input=x)    
        elif self.img_info['img_type'] == 'sar':
            x = self.add_GammaNoise(input=x, looks_of_sensor=self.img_info['looks_of_sensor'])
        # forward pass
        a1 = self.ae1(x)
        a2 = self.coupling_daes[0](a1)
        a3 = self.coupling_daes[1](a2)
        a4 = self.coupling_daes[2](a3)

        if self.training:
            return a1, a2, a3, a4 
        else:
            return a1, a2, a3, a4,  self.reconstruct(a4)
    
    def reconstruct(self, x):
        # reconstruct the input image
        a3_reconstruct = self.coupling_daes[2].reconstruct(x)
        a2_reconstruct = self.coupling_daes[1].reconstruct(a3_reconstruct)
        a1_reconstruct = self.coupling_daes[0].reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct



    
        


    

        



