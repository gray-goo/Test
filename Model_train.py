#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
import numpy as np
import h5py
import torch
from Model_define_pytorch import AutoEncoder, DatasetFolder,NMSE, Score
import os
import torch.nn as nn
import time
from sklearn.decomposition import PCA
import torchvision
import scipy.io as scio
from scheduler import WarmupMultiStepLR, WarmUpCosineAnnealingLR
import math

# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_single_gpu = True  # select whether using single gpu or multiple gpus
torch.manual_seed(1)
batch_size = 20
learning_rate = 1e-3
epochs = 100000
num_workers = 0
print_freq = 10  # print frequency (default: 60)
# parameters for data
feedback_bits = 512
exp_day='exp2_4'
exp_num='/2'
class_individual = 'A '


hdf5file = False


mat_train = '/Htrain.mat'
mat_var_train = 'H_train'
mat_test = '/Htest.mat'
mat_var_test = 'H_test'

if(class_individual =='A' ):
    Class_idx_train = '/idClassI_train.mat'
    var_Class_idx_train = 'idClassI_train'
    Class_idx_test= '/idClassI_test.mat'
    var_Class_idx_test = 'idClassI_test'
if (class_individual == 'B'):
    Class_idx_train = '/idClassII_train.mat'
    var_Class_idx_train = 'idClassII_train'
    Class_idx_test = '/idClassII_test.mat'
    var_Class_idx_test = 'idClassII_test'

# Class_B_idx_train= '/Class_B_idx_train.mat'
# var_Class_B_idx_train = 'Class_B_idx_train'
# Class_B_idx_test = '/Class_B_idx_test.mat'
# var_Class_B_idx_test = 'Class_B_idx_test'

# mat_train = '/Htrain_126_64.mat'
# mat_var_train = 'H_train_126_64'
# mat_test = '/Htest_126_64.mat'
# mat_var_test = 'H_test_126_64'

#
# classA = True
# if classA:
#     hdf5file = False
#     mat_train = '/Htrain_A.mat'
#     mat_var_train = 'Class_A_idx_train'
#     #mat_var_train = 'H_train'
#     mat_test = '/Htest_A.mat'
#     #mat_var_test = 'H_test_A'
#     mat_var_test = 'Class_A_idx_test'
# else:
#     hdf5file = True
#     #mat_train = '/Htrain_B_dup.mat'
#     mat_train = '/Htrain_B.mat'
#     mat_var_train = 'Class_B_idx_train'
#     mat_test = '/Htest_B.mat'
#     mat_var_test = 'Class_B_idx_test'

# Model construction
model = AutoEncoder(feedback_bits)
#model.encoder.load_state_dict(torch.load('./Modelsave/resnet18-5c106cde.pth'))

if use_single_gpu:
    model = model.cuda()
else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    autoencoder = torch.nn.DataParallel(model).cuda()

criterion = nn.MSELoss().cuda()
#criterion = nn.SmoothL1Loss(beta=0.1).cuda()

# d_model = 256
# learning_rate =  1/(d_model**0.5)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9,0.98), eps=1e-9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print('Start loading data')
data_load_address = './data'
# if hdf5file:
#     mat = h5py.File(data_load_address+mat_train)
#     x_train =np.array(mat[mat_var_train]).transpose()
# else:

#B idx
# mat = scio.loadmat(data_load_address+Class_B_idx_train)
# B_mat_train_idx = mat[var_Class_B_idx_train] -1  # shape=8000*126*128*2
# B_mat_train_idx = np.reshape(B_mat_train_idx, [-1])
#
# mat = scio.loadmat(data_load_address+Class_B_idx_test)
# B_mat_test_idx = mat[var_Class_B_idx_test] -1 # shape=8000*126*128*2
# B_mat_test_idx = np.reshape(B_mat_test_idx, [-1])

# print(np.shape(B_mat_train_idx))


# print(np.shape(B_mat_train_idx))
# print(B_mat_train_idx-1)

mat = scio.loadmat(data_load_address+mat_train)
x_train = mat[mat_var_train]  # shape=8000*126*128*2
print(np.shape(x_train))
x_train = np.transpose(x_train.astype('float32'),[0,3,1,2])
# x_train = x_train[B_mat_train_idx]
# print(np.shape(x_train))

mat = scio.loadmat(data_load_address+mat_test)
x_test = mat[mat_var_test]  # shape=2000*126*128*2
# x_test = x_test[B_mat_test_idx]
x_test = np.transpose(x_test.astype('float32'),[0,3,1,2])
print(np.shape(x_test))

if (class_individual =='A' or class_individual == 'B'):
    mat = scio.loadmat(data_load_address+Class_idx_train)
    mat_train_idx = mat[var_Class_idx_train] -1
    mat_train_idx = np.reshape(mat_train_idx, [-1])
    x_train = x_train[mat_train_idx]
    print(np.shape(x_train))
if (class_individual =='A' or class_individual == 'B'):
    mat = scio.loadmat(data_load_address+Class_idx_test)
    mat_test_idx = mat[var_Class_idx_test] -1
    mat_test_idx = np.reshape(mat_test_idx, [-1])
    x_test = x_test[mat_test_idx]
    print(np.shape(x_test))

# Data loading
# dataLoader for training
train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# LR sheduler
# lr_scheduler = WarmupMultiStepLR(optimizer, milestones=[20, 60, 100], gamma=0.1, warmup_iters=40, warmup_factor=1e-5)
# num_steps = len(train_loader) * epochs
# lr_scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer, T_max=num_steps, T_warmup=30 * len(train_loader), eta_min=5e-5)

# warmup_step = 40000
# def lr_lambda(step):
#     # return a multiplier instead of a learning rate
#     if step == 0 and warmup_step == 0:
#         return 1.
#     else:
#         return 1. / (step ** 0.5) if step > warmup_step \
#             else step / (warmup_step ** 1.5)

# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda,last_epoch=-1,verbose=False)

# model_path ="/home/zyz/python_project/Model_pytorch_2021_initial/Modelsave/exp2_1/7/990_0.00012161321081293863/encoder.pth.tar"
# model.encoder.load_state_dict(torch.load(model_path)['state_dict'])
#
# model_path = "/home/zyz/python_project/Model_pytorch_2021_initial/Modelsave/exp2_1/7/990_0.00012161321081293863/decoder.pth.tar"
# model.decoder.load_state_dict(torch.load(model_path)['state_dict'])

best_loss = 1
print('Start training')
st = time.time()
step = 0

for epoch in range(1, epochs+1):
    # model training
    model.train()
    for i, input in enumerate(train_loader):
        # adjust learning rate
        if epoch == 150:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * 0.3

        input = input.cuda()
        # compute output
        output = model(input)
        loss = criterion(output, input)#+0.5*criterion2(output, input)
        # compute gradient and do Adam step
        optimizer.zero_grad()


        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        step += 1

    if epoch % print_freq == 0:
        print('Epoch [{0}]: Training loss={loss:.8f}\t all time cost{time1} sec'.format(epoch, loss=loss.item(),time1=time.time()-st))
        # print("step: {step}\t learning rate = {lr}".format(step = step, lr = lr_scheduler.get_last_lr()))
    # if epoch % print_freq ==0:
        model.eval()
        total_loss = 0
        y_test = []

        with torch.no_grad():
            for i, input in enumerate(test_loader):
                input = input.cuda()
                output = model(input)
                output1 = output.cpu().numpy()
                if i == 0:
                    y_test = output1
                else:
                    y_test = np.concatenate((y_test, output1), axis=0)

                total_loss += criterion(output, input).item() * input.size(0)#+0.5*criterion2(output, input).item() * input.size(0)
            average_loss = total_loss / len(test_dataset)
            if average_loss < best_loss:
                # model save
                # save encoder
                #modelSave1 = './Modelsave/encoder_%s.pth.tar' % epoch
                #torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
                best_epoch = str(epoch)
                average_loss1 = str(average_loss)
                if not os.path.exists('./Modelsave/'+exp_day):
                    os.mkdir('./Modelsave/'+exp_day)
                if not os.path.exists('./Modelsave/'+exp_day+exp_num):
                    os.mkdir('./Modelsave/'+exp_day+exp_num)
                if not os.path.exists('./Modelsave/' + exp_day + exp_num + '/' + best_epoch+ '_'+average_loss1):
                    os.mkdir('./Modelsave/' + exp_day + exp_num + '/' + best_epoch+ '_'+average_loss1)
                modelSave1 = './Modelsave/' + exp_day + exp_num + '/' + best_epoch+ '_'+average_loss1 +'/encoder.pth.tar'
                torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
                # save decoder
                #modelSave2 = './Modelsave/decoder_%s.pth.tar'% epoch
                #torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
                modelSave2 = './Modelsave/' + exp_day + exp_num + '/' + best_epoch+ '_'+average_loss1 +'/decoder.pth.tar'
                torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
                print('Model saved \t'
                      'Epoch: [{0}]\t'
                      'Testing Loss current {loss:.8f}\t, previous loss {loss2:.8f}\t'.format(epoch, loss=average_loss, loss2=best_loss))
                best_loss = average_loss

                # score
                NMSE_test = NMSE(np.transpose(x_test, (0, 2, 3, 1)), np.transpose(y_test, (0, 2, 3, 1)))
                scr = Score(NMSE_test)
                if scr < 0:
                    scr = 0
                else:
                    scr = scr
                result = 'score=', str(scr)
                print('The NMSE is: {nmse}\t, score = {score}'.format(nmse=NMSE_test, score= scr))
            # else:
            #     static_epoch = static_epoch + 1

print('Training took', time.time()-st)

#
# for epoch in range(1, epochs+1):
#     # model training
#     model.train()
#     for i, input in enumerate(train_loader):
#         # adjust learning rate
#         if epoch == 200:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = learning_rate * 0.3
#         if epoch == 400:
#             for param_group in optimizer.param_groups:
#                param_group['lr'] = learning_rate * 0.1
#         if epoch == 1200:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = learning_rate * 0.05
#         input = input.cuda()
#         # compute output
#         output = model(input)
#         loss = criterion(output, input)#+0.5*criterion2(output, input)
#         # compute gradient and do Adam step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         #lr_scheduler.step()
#         #if i % print_freq == 0:
#         #    print('Epoch: [{0}][{1}/{2}]\t'
#         #          'Training Loss {loss:.6f}\t'.format(
#         #        epoch, i, len(train_loader), loss=loss.item()))
#     if epoch % print_freq == 0:
#         print('Epoch [{0}]: Training loss={loss:.8f}'.format(epoch, loss=loss.item()))
#     if epoch % 10 == 0:
#         model.eval()
#         total_loss = 0
#         total_mse = 0
#         with torch.no_grad():
#             for i, input in enumerate(test_loader):
#                 input = input.cuda()
#                 output = model(input)
#                 total_loss += criterion(output, input).item() * input.size(0)#+0.5*criterion2(output, input).item() * input.size(0)
#                 total_mse += np.sum(np.square(np.abs(output.cpu().numpy() - input.cpu().numpy())))
#             average_loss = total_loss / len(test_dataset)
#             mse = total_mse/len(test_dataset)/128/126/2
#             if average_loss < best_loss:
#                 # model save
#                 # save encoder
#                 #modelSave1 = './Modelsave/encoder_%s.pth.tar' % epoch
#                 #torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
#                 modelSave1 = './Modelsave/encoder.pth.tar'
#                 torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
#                 # save decoder
#                 #modelSave2 = './Modelsave/decoder_%s.pth.tar'% epoch
#                 #torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
#                 modelSave2 = './Modelsave/decoder.pth.tar'
#                 torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
#                 print("Model saved")
#                 print('Epoch: [{0}]\t'
#                       'Testing Loss current {loss:.8f}\t, previous loss {loss2:.8f}\t'.format(epoch, loss=average_loss, loss2=best_loss))
#                 best_loss = average_loss
#                 print('Testing MSE {mse:.8f}\t '.format(mse=mse))  # This equals to MSE Loss
#
# print('Training took', time.time()-st)