from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from PIL import Image
import models
import pickle
import sys
import time
from tensorboardX import SummaryWriter


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='mydataset',
                    help='training dataset (default: cifar100)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save',
                    default='./checkpoint_layer15_image64_cbam_depth_gap/',
                    type=str, metavar='PATH', help='path to save prune model (default: current directory)')
parser.add_argument('--toTensorform', type=int, default=0, help='0 for 0~1; 1 for -1~1; 2 for original type; 3 for normalization from original data; 4 for nomalization from toTensordata; 5 for (original-0.5)/0.5')

parser.add_argument('--arch', default='vgg_twopath', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('--subject', default='SubjectB', type=str, metavar='PATH',
                    help='subject to train')

print(torch.__version__)



args = parser.parse_args()

localtimenow = time.strftime("%Y%m%d%H%M", time.localtime())

#savefolder = args.save
savefolder = args.save+"/"+args.subject
#savefolder = args.save+"/"+args.subject+localtimenow
runs_path = savefolder+"/runs"

subject = args.subject
path_img = './RGBD_Numpy_mid_n30p160_noflip' #+ subject + '/x_img_train.npy'
path_label = './RGBD_Numpy_mid_n30p160_noflip' #+ subject + '/y_label_train.npy'

if not os.path.exists(runs_path):
    os.makedirs(runs_path)

# for tensorboardX
writer = SummaryWriter(runs_path)

args.cuda = not args.no_cuda and torch.cuda.is_available()

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(savefolder):
    os.makedirs(savefolder)


class MyDataset(Dataset):
    def __init__(self, path_img, path_label, train, subject, transform):
        if train:
            print('train')
            path_img = path_img + "/" + subject + '/x_img_train.npy'
            path_label = path_label + "/" + subject + '/y_label_train.npy'
        else:
            print('test')
            path_img = path_img + "/" + subject + '/x_img_test.npy'       
            path_label = path_label + "/" + subject + '/y_label_test.npy'
        print(path_img)
        img_RGBD = np.load(path_img)
        label_RGBD = np.load(path_label)
        print(len(label_RGBD))
        
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

        #print(img.shape, type(img)) # (64, 64, 4) <class 'numpy.ndarray'>
        #print(label, type(label)) # 2.0 <class 'numpy.float64'>

        if self.transform is not None:
            imgD = self.transform(imgD)
            imgRGB = self.transform(imgRGB)       
        if self.target_transform is not None:
            label = self.target_transform(label)

        #print(img.shape, type(img)) # torch.Size([4, 64, 64]) <class 'torch.FloatTensor'>
        #print(label, type(label)) # 2.0 <class 'numpy.float64'>

        return (imgRGB, imgD), label

    def __len__(self):
        return len(self.img_RGB)

print('start')
train_dataset = MyDataset(path_img, path_label, 1, subject, args.toTensorform)

test_dataset = MyDataset(path_img, path_label, 0, subject, args.toTensorform)

shuffle_dataset = True
random_seed = 42

dataset_size = len(train_dataset)
indices = list(range(dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices = indices
train_sampler = SubsetRandomSampler(train_indices)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
print(len(train_loader), '\n=============')

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
print(len(test_loader), '\n=============')

print(args.arch)
if args.refine:
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'] ,inputch=4)
    model.load_state_dict(checkpoint['state_dict'])
    cfg = checkpoint['cfg']
    print(cfg)
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth ,inputch=4)
    cfg = model.cfg

if args.cuda:
    model.cuda()

# epoch_step = 20
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=epoch_step, gamma=0.5)
#scheduler = lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
#scheduler = lr_scheduler.ExponentialLR(optimizer, 0.95)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1


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


def train(epoch):
    losses = AverageMeter()
    train_acc = AverageMeter()
    model.train()  # set module as training_mode

    for batch_idx, (data, target) in enumerate(train_loader):

        target = target.type(torch.LongTensor)
        data0 = data[0].type(torch.FloatTensor)
        data1 = data[1].type(torch.FloatTensor)
        
        if args.cuda:  # print(type(data), type(target)) # <class 'torch.FloatTensor'> <class 'torch.LongTensor'>
            data0, data1, target = data0.cuda(), data1.cuda(), target.cuda()

        data0, data1, target = Variable(data0), Variable(data1), Variable(target)
       
        optimizer.zero_grad()
        output = model(data0, data1)
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        loss.backward()

        if args.sr:
            updateBN()

        optimizer.step()

        train_correct = 0
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        train_temp = 100. * train_correct / target.size(0)
        n = data0.size(0)

        losses.update(loss.data[0], n)
        train_acc.update(train_temp, n)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tAverage Loss: {:.6f}'.format(
                epoch, batch_idx * len(data0), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), losses.val, losses.avg))
            # print(batch_idx, args.log_interval, len(data)) # 0 100 64...
            # 100 100 64...

            print('\t\t\t\t\tTraining Set Accuracy: {:.2f}\tTraining Set Average Accuracy: {:.2f}'.format(
                train_acc.val, train_acc.avg))

            # ============================= Tensorboard weight distribution =============================
            niter = epoch * len(train_loader) + batch_idx
            '''
            for name, param in model.named_parameters():
                name = name.replace('.', '/')
                writer.add_histogram(name, param.clone().cpu().data.numpy(), niter)
            '''

            writer.add_scalar('Train_Loss/Training_iter', losses.val, niter)
            writer.add_scalar('Train_Loss/Training_Avg_iter', losses.avg, niter)
            writer.add_scalar('Train_Accuracy/Training_iter', train_acc.val, niter)
            writer.add_scalar('Train_Accuracy/Training_Avg_iter', train_acc.avg, niter)


    writer.add_scalar('Train_Loss/Training_Avg_epoch', losses.avg, epoch)
    writer.add_scalar('Train_Accuracy/Training_epoch', train_acc.val, epoch)
    writer.add_scalar('Train_Accuracy/Training_Avg_epoch', train_acc.avg, epoch)

    return train_acc.avg




def test(epoch):
    losses = AverageMeter()
    test_acc = AverageMeter()
    model.eval()  # set module as evaluation

    for batch_idx, (data, target) in enumerate(test_loader):

        target = target.type(torch.LongTensor)
        data0 = data[0].type(torch.FloatTensor)
        data1 = data[1].type(torch.FloatTensor)

        if args.cuda:
            data0, data1, target = data0.cuda(), data1.cuda(), target.cuda()
        data0, data1, target = Variable(data0, volatile=True), Variable(data1, volatile=True), Variable(target)

        output = model(data0,data1)
        test_loss = criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        test_correct = 0
        test_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_temp = 100. * test_correct / target.size(0)
        n = data0.size(0)

        losses.update(test_loss.data[0], n)
        test_acc.update(test_temp, n)

        if batch_idx % args.log_interval == 0:
            niter = epoch * len(test_loader) + batch_idx
            # print(losses.val, losses.avg, test_acc.val, test_acc.avg)
            writer.add_scalar('Test_Loss/Testing_iter', losses.val, niter)
            writer.add_scalar('Test_Loss/Testing_Avg_iter', losses.avg, niter)
            writer.add_scalar('Test_Accuracy/Testing_iter', test_acc.val, niter)
            writer.add_scalar('Test_Accuracy/Testing_Avg_iter', test_acc.avg, niter)


    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        losses.avg, test_acc.sum/100., len(test_loader.dataset), test_acc.avg))

    writer.add_scalar('Test_Loss/Testing_Avg_epoch', losses.avg, epoch)
    writer.add_scalar('Test_Accuracy/Testing_Avg_epoch', test_acc.avg, epoch)

    return test_acc.avg


def save_checkpoint(state, is_best, filepath):
    """
    subpath = '/epoch' + str(epoch) + '/'
    path = filepath + subpath
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(state, os.path.join(path, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(path, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
    """
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))



best_prec_val = 0.
best_prec1 = 0.
lr = args.lr

def adjust_learning_rate(optimizer, epoch):
    """Change learning rate 0~79:0.1/80~119:0.01/120~160:0.001 (SGD)"""
    '''
    if epoch in [args.epochs * 0.5, args.epochs * 0.75]:  # [args.epochs*0.5, args.epochs*0.75]
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    '''
    # ================= Change learning rate 0~19:0.1/20~79:0.01/80~159:0.001 (SGD) =================
    '''
    if epoch in [args.epochs*0.125, args.epochs*0.5]: # [args.epochs*0.5, args.epochs*0.75]
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    '''

    '''
    if epoch in [args.epochs*0.02777, args.epochs*0.05]: # [10,25]
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch in [args.epochs*0.0555, args.epochs*0.08333]: # [55,70]
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    
    elif epoch in [args.epochs*0.25] # [45]
        lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch in [args.epochs*0.3, args.epochs*0.39]: # [55,70]
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif epoch in [args.epochs * 0.5]: # [90]
        lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch in [args.epochs * 0.56, args.epochs * 0.639]: # [100,115]
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif epoch in [args.epochs * 0.75]: # [135]
        lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch in [args.epochs * 0.806, args.epochs * 0.89]: # [145,160]
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return lr
    '''
    # ========= Sets the learning rate to the initial LR decayed by 2 every 30 epochs ========= #

    lr = args.lr * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


since1 = time.time()
for epoch in range(args.start_epoch, args.epochs):

    since = time.time()
    # ============================ Adjust_learning_rate ============================ #
    # adjust_learning_rate(optimizer, epoch)
    lr = adjust_learning_rate(optimizer, epoch)
    print(lr)
    writer.add_scalar('epoch/learning rate', lr, epoch)

    # ============================ Cosine_lr ============================
    '''
    scheduler.step()
    writer.add_scalar('epoch/learning rate', scheduler.get_lr()[0], epoch)
    '''

    prec_train = train(epoch)

    # counter for each epoch
    time_elapsed = time.time() - since
    print('Present epoch Training complete in {:.3f}s, {:.0f}m {:.0f}s'.format(
        time_elapsed, time_elapsed // 60, time_elapsed % 60))

    since2 = time.time()

    # prec_val = validate(epoch)
    '''
    writer.add_scalars('Train_Val_Accuracy/Train_Val_Avg_epoch', {'Train_acc':prec_train.item(),
                                                                  'Val_acc':prec_val.item()}, epoch)
    '''
    prec1 = test(epoch)

    time_elapsed2 = time.time() - since2
    print('\nPresent epoch Testing complete in {:.3f}s, {:.0f}m {:.0f}s\n'.format(
        time_elapsed2, time_elapsed2 // 60, time_elapsed2 % 60))

    #is_best_val = prec_val > best_prec_val
    #best_prec_val = max(prec_val, best_prec_val)
    
    
    is_best1 = prec1 > best_prec1
    
    if is_best1:
        best_prec1 = max(prec1, best_prec1)
    print("Best accuracy: "+str(best_prec1))
    save_checkpoint({
        'cfg': cfg,
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        #'best_prec_val': best_prec_val,
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        'inputch' : model.inputch,
    }, is_best1, filepath=savefolder)



time_elapsed1 = time.time() - since1
print('Total Training complete in {:.3f}s, {:.0f}m {:.0f}s'.format(
    time_elapsed1, time_elapsed1 // 60, time_elapsed1 % 60))

#print("Best validation accuracy : "+str(best_prec_val))
print("Best accuracy: "+str(best_prec1))

writer.export_scalars_to_json("./test.json")
writer.close()
