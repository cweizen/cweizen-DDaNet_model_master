import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
#from models import *
import torch.nn.functional as F
import models
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib import pyplot as plt
import math
from scipy import misc
from PIL import Image
#import cv2

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='mydataset',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=15,
                    help='depth of the vgg')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./test', type=str, metavar='PATHSAVE',
                    help='path to save the model (default: none)')
parser.add_argument('--datapath', default='./RGBD_Numpy_mid_n30p160_noflip', type=str, metavar='PATH',
                    help='subject to train')
parser.add_argument('--subject', default='SubjectA', type=str, metavar='PATH',
                    help='subject to train')
parser.add_argument('--normalization', type=int, default=1,
                    help='0 not normalization others normalization')
parser.add_argument('--topkranking', type=int, default=1,
                    help='for top k ranking')
parser.add_argument('--toTensorform', type=int, default=0, help='0 for 0~1; 1 for -1~1; 2 for original type; 3 for normalization from original data; 4 for nomalization from toTensordata; 5 for (original-0.5)/0.5')
parser.add_argument('--GAP', type=int, default=1, help='0 for false other for true')
parser.add_argument('--arch', default='vgg_cbam_depth_gap', type=str,
                    help='architecture to use')



args = parser.parse_args()

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def get_target_module(self, module_name, model):
        for name, module in self.model._modules.items():
            if name == module_name:
                return module

    def __call__(self, cx, dx):
        outputs = []
        self.gradients = []
       
        
        module = self.get_target_module('featurec1', self.model)
        cx = module(cx)
        module = self.get_target_module('featured1', self.model)
        dx = module(dx)
        module = self.get_target_module('featureb1', self.model)
        scale = module(dx)
        
        
        cx = cx * scale
        dx = dx * scale
        
        
        module = self.get_target_module('featurec2', self.model)
        cx = module(cx)
        module = self.get_target_module('featured2', self.model)
        dx = module(dx)
        module = self.get_target_module('featureb2', self.model)
        scale = module(dx)
        
        
        cx = cx * scale
        dx = dx * scale
        
        
        module = self.get_target_module('featurec3', self.model)
        cx = module(cx)
        module = self.get_target_module('featured3', self.model)
        dx = module(dx)
        module = self.get_target_module('featureb3', self.model)
        scale = module(dx)

        ## get this mask
        outputs = scale
        
        cx = cx * scale
        dx = dx * scale 
        

        module = self.get_target_module('featurec4', self.model)
        cx = module(cx)


        x = torch.cat((cx, dx),1)


        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		# for GAP
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.feature_extractor = FeatureExtractor(self.model, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, cx ,dx):
                target_activations, output  = self.feature_extractor(cx, dx)
                output = self.avgpool(output) 
                output = output.view(output.size(0), -1)
                output = self.model.classifier(output)
                return target_activations, output


class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input_color, input_depth):
		return self.model(input_color, input_depth) 

	def __call__(self, input_color, input_depth, index = None):
		if self.cuda:
			features, output = self.extractor(input_color.cuda(), input_depth.cuda())
		else:
			features, output = self.extractor(input_color, input_depth)
		print(features)
		cam = features.cpu().data.numpy()[0, 0, :]
		
		#cam = np.maximum(cam, 0)
		print(cam)
		print(cam.shape)

		#cam = cv2.resize(cam, (64, 64))
		cam = misc.imresize(cam, (64, 64))
		cam = cam - np.min(cam)
		if np.max(cam) != 0:
			cam = cam / np.max(cam)
		return cam

print(torch.__version__)


args.cuda = not args.no_cuda and torch.cuda.is_available()
path_img_test = args.datapath+'/'+args.subject+'/x_img_test.npy'
path_label_test = args.datapath+'/'+args.subject+'/y_label_test.npy'
topk = args.topkranking

if args.topkranking<=0 or args.topkranking>=25:
    topk = 1

print(args.arch)
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        
        checkpoint = torch.load(args.model)
        
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth ,inputch=4, cfg=checkpoint['cfg'])
        # print(checkpoint['cfg'])
        # print(cfg)

        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}\n"
              .format(args.model, checkpoint['epoch'], best_prec1))
        #print(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'\n".format(args.resume))


if args.cuda:
    model.cuda()




class MyDataset(Dataset):
    def __init__(self, path_img, path_label, transform=0):
        
        img_RGBD = np.load(path_img)
        label_RGBD = np.load(path_label)

        #'0 for 0~1; 1 for -1~1; 2 for original type; 3 for normalization from original data; 4 for nomalization from toTensordata; 5 for (original-0.5)/0.5'
        if transform == 0 or transform ==1 or transform ==4:
            self.img_RGBD = img_RGBD.astype(np.uint8)
        else:
            self.img_RGBD = img_RGBD
        self.label_RGBD = label_RGBD
        
        # in this test: train and test mean or std calculate seperately (need to change in the future)
        if transform ==1 or transform==5:
            mean = [0.5,0.5,0.5,0.5]
            std = [0.5,0.5,0.5,0.5]
        elif transform == 3:
            mean = [np.mean(self.img_RGBD[:,:,:,0]),np.mean(self.img_RGBD[:,:,:,1]),np.mean(self.img_RGBD[:,:,:,2]),np.mean(self.img_RGBD[:,:,:,3])]
            std = [np.std(self.img_RGBD[:,:,:,0]),np.std(self.img_RGBD[:,:,:,1]),np.std(self.img_RGBD[:,:,:,2]),np.std(self.img_RGBD[:,:,:,3])]
        elif transform == 4:
            mean = [np.mean(self.img_RGBD[:,:,:,0])/(256),np.mean(self.img_RGBD[:,:,:,1])/(256),np.mean(self.img_RGBD[:,:,:,2])/(256),np.mean(self.img_RGBD[:,:,:,3])/(256)]
            std = [np.std(self.img_RGBD[:,:,:,0])/(256*256),np.std(self.img_RGBD[:,:,:,1])/(256*256),np.std(self.img_RGBD[:,:,:,2])/(256*256),np.std(self.img_RGBD[:,:,:,3])/(256*256)]
        else:
            mean = [0,0,0,0]
            std = [1,1,1,1]

        self.transform =  transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((mean[0], mean[1], mean[2], mean[3]), (std[0], std[1], std[2], std[3]))])
        self.target_transform = None
        

        self.img_RGB = img_RGBD[:,:,:,0:3]
        img_D = img_RGBD[:,:,:,3]
        self.img_D = img_D[:,:,:,np.newaxis]
        self.label_RGBD = label_RGBD

    def __getitem__(self, index):

        imgRGB, imgD, label = self.img_RGB[index], self.img_D[index], self.label_RGBD[index]


        imgRGB_o = imgRGB.copy()
        imgD_o = imgD.copy()

        #print(img.shape, type(img)) # (64, 64, 4) <class 'numpy.ndarray'>
        #print(label, type(label)) # 2.0 <class 'numpy.float64'>

        if self.transform is not None:
            imgD = self.transform(imgD)
            imgRGB = self.transform(imgRGB)       
        if self.target_transform is not None:
            label = self.target_transform(label)
       
        #print(img.shape, type(img)) # torch.Size([4, 64, 64]) <class 'torch.FloatTensor'>
        #print(label, type(label)) # 2.0 <class 'numpy.float64'>

        return (imgRGB_o, imgD_o), (imgRGB, imgD), label

    def __len__(self):
        return len(self.img_RGB)

# Top-3
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n = 1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model,topk):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mydataset':
        test_loader = torch.utils.data.DataLoader(
            MyDataset(path_img_test, path_label_test, transform = args.toTensorform),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        # print(len(test_loader)) # 50

    # Top-3
    
    top3 = AverageMeter()
    top5 = AverageMeter()
   

    model.eval()
    correct = 0
    pred_result = np.zeros(0)
    target_result = np.zeros(0)
    ierror = 0
    batchindex = 0
    numoforginaldata = np.zeros((1,25))

    grad_cam = GradCam(model = model, target_layer_names = ["38"], use_cuda=args.cuda)

    
    for ordata, data, target in test_loader:
        # print(type(data), type(target))
        
        data_RGB, data_D = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor)
        target = target.type(torch.LongTensor)
        target1 = target.cpu().numpy()

        if args.cuda:
            data_RGB, data_D, target = data_RGB.cuda(), data_D.cuda(), target.cuda()

        undata_RGB, undata_D, testdata_RGB, testdata_D, target = Variable(data_RGB, volatile=False), Variable(data_D, volatile=False), Variable(data_RGB, volatile=True), Variable(data_D, volatile=True), Variable(target)
        output = model(testdata_RGB, testdata_D)  # <class 'torch.autograd.variable.Variable'> torch.Size([256, 25])
        
        # print(output[1])
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # array = [0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24];=
        arrayindex = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
        #print(torch.unsqueeze(data[0],0).shape)
        
        for inum in range(len(pred)):
            nofdata = 256*batchindex+inum;
            ## nofdata: choose pictures by number; int(target[inum]) choose piture by target letter
            if int(target[inum])==4 or int(target[inum])==10 or int(target[inum])==11 or int(target[inum])==15 or int(target[inum])==16 or int(target[inum])==18 or int(target[inum])==19 or int(target[inum])==23:# r 17 e 4 p 15 n 13 w 22
                # original data
                ordata_RGB, ordata_D = ordata[0], ordata[1]
                Odata = ordata_D[inum].numpy()   
                Ddata = np.zeros((Odata.shape[0],Odata.shape[1]),dtype = np.uint8)
                
                for ix in range(Odata.shape[0]):
                    for iy in range(Odata.shape[1]):
                            Ddata[ix,iy]=int(Odata[ix,iy,0])
                
                # generate original data for original image
                #im = Image.fromarray(RGBdata,'RGB')
                if int(pred[inum])!=int(target[inum]):
		    #pred_target_pred          
                    savestream = (args.save+"/"+args.subject+"/pred_"+arrayindex[int(target[inum])]+"_"+arrayindex[int(pred[inum])])
		    # print(savestream)
                    
                    if not os.path.exists(savestream):
                        os.makedirs(savestream)
                    

                    # get heat-map for pred

                    dimsoftmax = nn.Softmax()
                    newsoft = dimsoftmax(output[inum]).data.cpu().numpy()
                    # get pred cam
                    input_RGB, input_D = torch.unsqueeze(undata_RGB[inum],0).float(), torch.unsqueeze(undata_D[inum],0).float()
                    target_index = int(pred[inum])
                    heat_map = grad_cam(input_RGB, input_D, target_index)

                    plt.imshow(heat_map,cmap='gray')
                    #plt.imshow(heat_map,cmap='jet',alpha=0.4)
                    plt.title("("+arrayindex[int(target[inum])]+","+arrayindex[int(pred[inum])]+","+arrayindex[int(pred[inum])]+")")
                    ##                    
                    # ensure the folder is exist
                    if not os.path.exists(savestream+"/pred_"+arrayindex[int(pred[inum])]+"/"):
                        os.makedirs(savestream+"/pred_"+arrayindex[int(pred[inum])]+"/")
                    plt.savefig(savestream+"/pred_"+arrayindex[int(pred[inum])]+"/"+str(256*batchindex+inum)+"_"+str(round(newsoft[int(pred[inum])],5))+".png")
                    
                    
                    # get target cam
                    input_RGB, input_D = torch.unsqueeze(undata_RGB[inum],0).float(), torch.unsqueeze(undata_D[inum],0).float()
                    target_index = int(target[inum])
                    heat_map = grad_cam(input_RGB, input_D, target_index)

                    plt.imshow(heat_map,cmap='gray')
                    plt.title("("+arrayindex[int(target[inum])]+","+arrayindex[int(pred[inum])]+","+arrayindex[int(target[inum])]+")")
                    ##
                      
                    # ensure the folder is exist
                    if not os.path.exists(savestream+"/target_"+arrayindex[int(target[inum])]+"/"):
                        os.makedirs(savestream+"/target_"+arrayindex[int(target[inum])]+"/")
                    plt.savefig(savestream+"/target_"+arrayindex[int(target[inum])]+"/"+str(256*batchindex+inum)+"_"+str(round(newsoft[int(target[inum])],5))+".png")   
  
                elif int(pred[inum])==int(target[inum]):
                    ## number of samples for correct results
                    if numoforginaldata[0,int(target[inum])]<=1000:
                        savestream = (args.save+"/"+args.subject+"/pred_"+arrayindex[int(target[inum])]+"_"+arrayindex[int(pred[inum])])
		        # print(savestream)
                        
                        if not os.path.exists(savestream):
                            os.makedirs(savestream)

                        dimsoftmax = nn.Softmax()
		        # print(dimsoftmax(output[inum]))   
		        #get sort index 
                    
                        newsoft = dimsoftmax(output[inum]).data.cpu().numpy()
                        
                        # save cam
                        input_RGB, input_D = torch.unsqueeze(undata_RGB[inum],0).float(), torch.unsqueeze(undata_D[inum],0).float()
                        target_index = int(target[inum])
                        heat_map = grad_cam(input_RGB, input_D, target_index)

                        plt.imshow(heat_map,cmap='gray')
                        #plt.imshow(heat_map,cmap='jet',alpha=0.4)
                        plt.title("("+arrayindex[int(target[inum])]+","+arrayindex[int(pred[inum])]+","+arrayindex[int(target[inum])]+")")
                        plt.savefig(savestream+"/"+str(256*batchindex+inum)+"_"+str(round(newsoft[int(target[inum])],5))+".png")
                        ##
                        # take 10 correct images for ecah item
                        numoforginaldata[0,int(target[inum])] = numoforginaldata[0,int(target[inum])]+1
                        
        batchindex = batchindex+1
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # view_as : distribute new tensor for target.data which have the same tensor size as "pred"

        # Top-3 Top-5
        
        prec3, prec5 = accuracy(output, target, topk=(3, 5))
        n = data_RGB.size(0)
        top3.update(prec3.data[0], n)
        top5.update(prec5.data[0], n)
        
        

        pred = torch.squeeze(pred)
        pred = pred.cpu().numpy()
        pred_result = np.concatenate((pred_result, pred))  # <class 'numpy.ndarray'> (12547,)
        pred_result = pred_result.astype(int)

        target_result = np.concatenate((target_result, target1))  # <class 'numpy.ndarray'> (12547,)
        target_result = target_result.astype(int)

  
    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
          correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    print("top1: ",100. * correct / len(test_loader.dataset), "top3: ",top3.avg, "top5: ", top5.avg)
    # print("target:",target_result,"pred :", pred_result)
    return target_result, pred_result

def plot_histgram_score(x_score,savepath):
    arrayindex = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
    # x_score size 25*10
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for i in range(len(x_score)):
        X = np.arange(len(x_score[i]))
        plt.bar(X,x_score[i])
        for x,y in zip(X,x_score[i]):
            plt.text(x+0.05,y+0.05,'%.0f' %y, ha='center', va='bottom')
        plt.savefig(savepath+"/"+str(arrayindex[i])+'.png')
        plt.close()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'


    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)  # <class 'numpy.ndarray'> (24, 24)

    # print(cm)

    if normalize:
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label', fontsize=16)

    # ax.xaxis.label.set_size(14)
    # ax.yaxis.label.set_size(20)

    # Rotate the tick labels and set their alignment.
    """
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    """

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    # fmt1 = '.0f' if normalize else 'd'

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.0f' if cm[i,j]<0.01 else fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                    ,fontsize=7)

    plt.tick_params(labelsize=14)
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)


y_test, y_pred = test(model,topk)

# array = [a b c d e f g h i k  l  m  n  o  p  q  r  s  t  u  v  w  x  y ]
# array = [0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
class_names = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'])
'''
# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=args.normalization, title=None)  
# Confusion matrix, without normalization
# Normalized confusion matrix
plt.show()
'''


