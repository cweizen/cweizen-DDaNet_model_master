import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import ceil,floor
from .CBAM_DAM import *


__all__ = ['vgg_cbam_depth_gap']


# 'M':maxpooling 'S':spatial attention 'C':channel attention 'X':spatial and channel attention
defaultcfg = {
    15 : ['a1', 64, 64, 'BS', 'M', 128, 128, 'b1', 'DS', 'a2', 'CS', 'M', 256, 256, 256, 256, 'b2', 'DS', 'a3', 'CC', 'M', 512, 512, 512, 512, 'b3', 'DS', 'a4', 'CC'],
}


class vgg_cbam_depth_gap(nn.Module):
    def __init__(self, dataset='mydataset', depth=15, init_weights=True, cfg=None , inputch=3, usingGAP=False ,hyper_r=1, lastactivation=None):
        super(vgg_cbam_depth_gap, self).__init__()
        self.inputch = inputch
        self.lastactivation = lastactivation
        #self.noiselayer = DynamicGNoise(64)
        if cfg is None:
            cfg = defaultcfg[depth]
            print(cfg)
        self.cfg = cfg
        self.featurec1, self.featurec2, self.featurec3, self.featurec4, self.featured1, self.featured2, self.featured3, self.featureb1, self.featureb2, self.featureb3 = self.make_layers(cfg, True)
        self.usingGAP = usingGAP
        self.avgpoolingcolor = nn.AdaptiveAvgPool2d(1)
        self.avgpoolingdepth = nn.AdaptiveAvgPool2d(1)
      
        if dataset == 'mydataset':
            num_classes = 25
        self.classifier = nn.Linear(512*2, num_classes)
	
        # Multiple FC layer design
        '''
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        '''

        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layersc1 = []
        layersc2 = []
        layersc3 = []
        layersc4 = []
        layersd1 = []
        layersd2 = []
        layersd3 = []
        layersb1 = []
        layersb2 = []
        layersb3 = []

        layersc = []
        layersd = []
        layersb = []
        
        
        flag = 0
        wflag = 'a'
        for v in cfg:
            if v == 'a1':
                flag = 1
                wflag = 'a'
                in_channelsc = 3
                in_channelsd = 1
            elif v == 'b1':
                layersc1 = layersc
                layersd1 = layersd
                layersc = []
                layersd = []
                flag = 1
                wflag = 'b'
            elif v == 'a2':
                layersb1 = layersb
                layersb = []
                flag = 2
                wflag = 'a'
            elif v == 'b2':
                layersc2 = layersc
                layersd2 = layersd
                layersc = []
                layersd = []
                flag = 2
                wflag = 'b'
            elif v == 'a3':
                layersb2 = layersb
                layersb = []
                flag = 3
                wflag = 'a'
            elif v == 'b3':
                layersc3 = layersc
                layersd3 = layersd
                layersc = []
                layersd = []
                flag = 3
                wflag = 'b'
            elif v == 'a4':
                layersb3 = layersb
                layersb = []
                flag = 4
                wflag = 'a'
            elif v == 'M':
                layersc += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layersd += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'BS':
                layersc += [SpatialGate()]
                layersd += [SpatialGate()]
            elif v == 'CS':
                layersc += [SpatialGate()]
            elif v == 'DS':
                layersb += [SpatialGate_depth()]
            elif v == 'CC':
                layersc += [ChannelGate(in_channelsc)] 
            else:
                conv2dc = nn.Conv2d(in_channelsc, v, kernel_size=3, padding=1, bias=False)
                conv2dd = nn.Conv2d(in_channelsd, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layersc += [conv2dc, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    layersd += [conv2dd, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    # layers += [conv2d, nn.BatchNorm2d(v), nn.Tanh()]
                else:
                    layersc += [conv2dc, nn.ReLU(inplace=True)]
                    layersd += [conv2dd, nn.ReLU(inplace=True)]
                    # layers += [conv2d, nn.Tanh()]
                in_channelsc = v
                in_channelsd = v
        layersc4 = layersc


        return nn.Sequential(*layersc1), nn.Sequential(*layersc2), nn.Sequential(*layersc3), nn.Sequential(*layersc4), nn.Sequential(*layersd1), nn.Sequential(*layersd2), nn.Sequential(*layersd3), nn.Sequential(*layersb1), nn.Sequential(*layersb2), nn.Sequential(*layersb3)

    def forward(self, cx, dx):  # output = model(data) --> Start work
        # print("BatchInput_x : {}\n".format(x.shape)) # BatchInput_x: torch.Size([64, 3, 64, 64])
        cx = self.featurec1(cx)
        dx = self.featured1(dx)
        scale = self.featureb1(dx)
        cx = cx * scale
        dx = dx * scale
        cx = self.featurec2(cx)
        dx = self.featured2(dx)  
        scale = self.featureb2(dx)
        cx = cx * scale
        dx = dx * scale
        cx = self.featurec3(cx)
        dx = self.featured3(dx)
        scale = self.featureb3(dx)
        cx = cx * scale
        dx = dx * scale
        cx = self.featurec4(cx)

        '''
        print(xa.size())
        print(xb.size())
        print(xc.size())
        '''
        cx = self.avgpoolingcolor(cx)
        dx = self.avgpoolingdepth(dx)
        
        x = torch.cat((cx,dx),1)
        x = x.view(x.size(0), -1)
        
        y = self.classifier(x)
        # print(type(y), y)

        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5) # Gamma ---I guess
                m.bias.data.zero_() # Beta ---I guess
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    #x=>input i=>object
    '''
    def featuremap_cam(self, x):  # output = model(data) --> Start work
        # print("BatchInput_x : {}\n".format(x.shape)) # BatchInput_x: torch.Size([64, 3, 64, 64])
   
        x0 = self.feature0(x)
        xa = self.featurea(x0)
        x1 = self.feature1(x0)
        xb = self.featureb(x1)
        x2 = self.feature2(x1)
        xc = self.featurec(x2)
        x3 = self.feature3(x2)
        xd = self.featured(x3)
        
        
        x = torch.cat((xa,xb,xc,xd),1)
        x = self.featurelast(x)
        x = self.avgpooling(x)

        x = x.view(x.size(0), -1)      
 
        y = x
        
        # print(type(y), y)

        return y
    '''
    def get_gap(self,x,i):
        # print("BatchInput_x : {}\n".format(x.shape)) # BatchInput_x: torch.Size([64, 3, 64, 64])

        x = self.feature(x)
        # print("feature_x : {}\n".format(x.shape)) # feature_x: torch.Size([64, 112, 4, 4])
        # get feature map before gap
        GAP_x = x
        # concatenation 512 channel feature map with weight
        self.GAP_x = np.zeros((GAP_x.shape[0],1,GAP_x.shape[2],GAP_x.shape[3]),dtype = float)
        weight_shape = self.classifier.weight.data.shape
        weight_data = self.classifier.weight.data.cpu().numpy()
        # cam start
        print('cam start')
        print(GAP_x.shape[0],GAP_x.shape[1],GAP_x.shape[2],GAP_x.shape[3])
        print(len(weight_data[i,:]))
        for cam_b in range(GAP_x.shape[0]):
            for wi in range(len(weight_data[i,:])):
                if wi%4 ==0:
                    archor_y,archor_x = 0,0
                    for dx in range(ceil(GAP_x.shape[3]/2)):
                        for dy in range(ceil(GAP_x.shape[2]/2)):
                            self.GAP_x[cam_b,0,archor_y+dy,archor_x+dx]=weight_data[i,wi]*GAP_x[cam_b,wi//4,archor_y+dy,archor_x+dx]+self.GAP_x[cam_b,0,archor_y+dy,archor_x+dx]
                elif wi%4 ==1:
                    archor_y,archor_x = 0,ceil(GAP_x.shape[3]/2)
                    for dx in range(floor(GAP_x.shape[3]/2)):
                        for dy in range(ceil(GAP_x.shape[2]/2)):
                            self.GAP_x[cam_b,0,archor_y+dy,archor_x+dx]=weight_data[i,wi]*GAP_x[cam_b,wi//4,archor_y+dy,archor_x+dx]+self.GAP_x[cam_b,0,archor_y+dy,archor_x+dx]
                elif wi%4 ==2:
                    archor_y,archor_x = ceil(GAP_x.shape[2]/2),0
                    for dx in range(ceil(GAP_x.shape[3]/2)):
                        for dy in range(floor(GAP_x.shape[2]/2)):
                            self.GAP_x[cam_b,0,archor_y+dy,archor_x+dx]=weight_data[i,wi]*GAP_x[cam_b,wi//4,archor_y+dy,archor_x+dx]+self.GAP_x[cam_b,0,archor_y+dy,archor_x+dx]
                elif wi%4 ==3:
                    archor_y,archor_x = ceil(GAP_x.shape[2]/2),ceil(GAP_x.shape[3]/2)
                    for dx in range(floor(GAP_x.shape[3]/2)):
                        for dy in range(floor(GAP_x.shape[2]/2)):
                            self.GAP_x[cam_b,0,archor_y+dy,archor_x+dx]=weight_data[i,wi]*GAP_x[cam_b,wi//4,archor_y+dy,archor_x+dx]+self.GAP_x[cam_b,0,archor_y+dy,archor_x+dx]
        for cam_b in range(GAP_x.shape[0]):
            self.GAP_x[cam_b,:,:,:] = self.GAP_x[cam_b,:,:,:]-np.min(self.GAP_x[cam_b,:,:,:])
            self.GAP_x[cam_b,:,:,:] = self.GAP_x[cam_b,:,:,:]/np.max(self.GAP_x[cam_b,:,:,:])
        
        return self.GAP_x
class DynamicGNoise(nn.Module):
    def __init__(self, shape, std=0.05):
        super().__init__()
        self.noise = Variable(torch.zeros(shape,shape).cuda())
        self.std   = std
        
    def forward(self, x):
        if not self.training: 
            return x
        self.noise.data.normal_(0, std=self.std)
        '''       
        print(self.noise)
        print(x.size(), self.noise.expand_as(x).size())
        print(self.noise.expand_as(x))
        '''
        return x + self.noise.expand_as(x)

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.avgpool = nn.AdaptiveAvgPool2d(2)
		self.feature_extractor = FeatureExtractor(self.model.feature, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
                target_activations, output  = self.feature_extractor(x)
                output = self.avgpool(output) 
                output = output.view(output.size(0), -1)
                output = self.model.classifier(output)
                return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("cam.jpg", np.uint8(255 * cam))

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)
		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.feature.zero_grad()
		self.model.classifier.zero_grad()
		one_hot.backward(retain_variables=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (64, 64))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

if __name__ == '__main__':
    net = vgg_cbam_depth_gap(usingGAP=True)
    #x = Variable(torch.FloatTensor(16, 3, 40, 40))
    #print(net.inputch)
    #x = Variable(torch.FloatTensor(1, 4, 64, 64))
    cx = Variable(torch.zeros([1,3,64,64]))
    dx = Variable(torch.zeros([1,1,64,64]))
    #x = Variable(torch.zeros([1,4,80,80]))
    cx[0,0,0,0]=1
    cx[0,0,63,63]=1
    dx[0,0,0,10]=1
    dx[0,0,63,30]=1
    y = net(cx,dx)

    '''
    input = x
    grad_cam = GradCam(model = net, target_layer_names = ["50"], use_cuda=True)
    target_index = 2
    mask = grad_cam(input, target_index)
    plt.imshow(mask)
    plt.show()
    '''
    numofparameters = sum(param.numel() for param in net.parameters())/1000000
    print(numofparameters)
    print(y.data.shape)
