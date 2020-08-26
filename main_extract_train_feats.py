from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import progress_bar, adjust_learning_rate
from datasets import load_path_feat_extraction

from torchvision import transforms

import scipy.io as sio

import pdb, os
import pandas as pd
import h5py



import torch.nn as nn

from dirtorch.utils import common
import dirtorch.nets as nets

import numpy as np



############################
#
# Training and Testing
#
############################
def feature_extraction_(train_test_loader, eval_data_type):
	net.eval()
	with torch.no_grad():

		total_samples = len(train_test_loader.dataset)		
		X_ = torch.zeros([len(train_test_loader.dataset), 2048])
		# labels = torch.zeros([len(train_test_loader.dataset), 1])
		# gt_class_labels = torch.zeros([len(train_test_loader.dataset), 1])
		pid = []
		cid =[]
		k=0
		
		for batch_idx, (data, pid_targets, cid_targets) in enumerate(train_test_loader):

			print('[{}/{}]'.format(batch_idx,len(train_test_loader)))	
			
			# Data and labels
			data = data.to(device)
			
			x_ = net(data)

			# Trained features
			X_[k*batch_size:(k*batch_size)+data.shape[0],:] = x_.view(data.shape[0],x_.shape[1])
			# labels[k*batch_size:(k*batch_size)+data.shape[0]] = targets
			# gt_class_labels[k*batch_size:(k*batch_size)+data.shape[0]] = class_labels

			pid = np.concatenate((pid,pid_targets[0]))
			cid = np.concatenate((cid,cid_targets[0]))

			k+=1
			

		X_ 	= X_.cpu().detach().numpy()
		# labels 	= labels.cpu().detach().numpy()
		# class_labels = gt_class_labels.cpu().detach().numpy()

		output_dir = 'features/ClusteringFeats/BASE/{}'.format(features_output_file_path)
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)

		with h5py.File('{}/train_{}_{}_{}.h5'.format(output_dir,eval_data_type,layer_name,load_epoch_num+1), 'w') as hf:
			hf.create_dataset('X_', data=X_, dtype='float32')
			hf.create_dataset('pid', data=pid, dtype='int')
			hf.create_dataset('cid', data=cid, dtype='int')
		

def load_model(path, iscuda):
	checkpoint = common.load_checkpoint(path, iscuda)
	net = nets.create_model(pretrained="", **checkpoint['model_options'])
	net = common.switch_model_to_cuda(net, iscuda, checkpoint)
	net.load_state_dict(checkpoint['state_dict'])
	net.preprocess = checkpoint.get('preprocess', net.preprocess)
	if 'pca' in checkpoint:
		# print('PCA loaded!')
		net.pca = checkpoint.get('pca')
	return net


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='KIT NLE Fashion Project')
	parser.add_argument('--batch-size', type=int, default=60, metavar='N',
											help='Batch size')
	parser.add_argument('--load-epoch', type=int, default=45, metavar='N',
											help='load epoch number')
	parser.add_argument('--optimizer', type=str, default='ADAM',
											help='ADM | SGD')
	parser.add_argument('--df-comb', type=str, default='ALL',
											help='ALL | C2S')
	parser.add_argument('--s2s-comb', type=str, default='XXX',
											help='C2C | S2S')
	parser.add_argument('--model-name', type=str, default='DeepFashion',
											help='DeepFashion | DF_S2S ')
	parser.add_argument('--eval-dataset', type=str, default='XXX',
											help='DeepFashion | Street2Shop ')
	parser.add_argument('--resume', '-r', action='store_true')
	parser.add_argument('--layer', type=str, default='X')
	parser.add_argument('--checkpoint', type=str, default='../dirtorch/data/Resnet101-TL-GeM.pt',
											 help='path to weights')

	args = parser.parse_args()

	#############################
	#
	# Setting default parameters
	#
	#############################

	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	batch_size = args.batch_size
	model_name = args.model_name

	combinations_type = args.df_comb #'ALL' # Consumer2Shop(C2S), Shop2Shop(S2S), Consumer2Consumer(C2C), ALL: (C2S, C2C, S2S)

	checkpoint_path = 'models'
	train_test_type = 'test' 
	eval_dataset_name = args.eval_dataset
	layer_name = args.layer

	#############################
	#
	# Paths
	#
	#############################

	if eval_dataset_name == 'DeepFashion':
		path_to_images_ = '/cvhci/data/fashion_NAVER/DeepFashion/Consumer-to-shop_Clothes_Retrieval_Benchmark/Consumer-to-shop_Clothes_Retrieval_Benchmark/cropped_img'
		path_to_consumer_ = 'dataset_files/train_cluster/df_consumer.txt'
		path_to_shop_ = 'dataset_files/train_cluster/df_shop.txt'

	elif eval_dataset_name == 'Street2Shop':
		path_to_images_ = '/cvhci/data/fashion_NAVER/Street2Shop/s2s_cropped/detX101_Jan19/img'
		path_to_consumer_ = 'dataset_files/train_cluster/s2s_consumer.txt'
		path_to_shop_ = 'dataset_files/train_cluster/s2s_shop.txt'

	#############################
	#
	# Dataloader
	#
	#############################
	# net.preprocess
	#  {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'input_size': 224}
	output_size_ = 224
	transform_test = transforms.Compose([
		transforms.Resize([output_size_, output_size_]),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	consumer_set = load_path_feat_extraction(path_to_images_=path_to_images_, path_to_txt_=path_to_consumer_,transform=transform_test)
	shop_set = load_path_feat_extraction(path_to_images_=path_to_images_, path_to_txt_=path_to_shop_,transform=transform_test)

	# Data loader
	consumer_loader = torch.utils.data.DataLoader(dataset=consumer_set,
											 batch_size=batch_size,
											 shuffle=False,
											 num_workers=8)

	shop_loader = torch.utils.data.DataLoader(dataset=shop_set,
											 batch_size=batch_size,
											 shuffle=False,
											 num_workers=8)

	#############################
	#
	# Model
	#
	#############################

	net = load_model(args.checkpoint, 'cpu')

	######################
	# For loading models
	#
	#####################
	if model_name =='DeepFashion':
		checkpoint_file_name = '{}/{}/{}_{}_{}'.format(checkpoint_path,model_name,model_name,args.optimizer,combinations_type)
		features_output_file_path = '{}/{}_{}_{}'.format(eval_dataset_name,model_name,args.optimizer,combinations_type)


	######################
	# For storing features
	#
	######################

	load_epoch_num = args.load_epoch - 1 

	if args.resume:
		# Load checkpoint.
		checkpoint_number = '{}_{}_ckpt.t7'.format(checkpoint_file_name,load_epoch_num)
		print(checkpoint_number)
		print('==> Resuming from checkpoint..')
		assert os.path.isdir('models'), 'Error: no checkpoint directory found!'
		checkpoint = torch.load(checkpoint_number)
		net.load_state_dict(checkpoint['state_dict'])
		net =  net.to(device)

	if device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True




	#############################
	#
	# Model evaluation 
	#
	#############################
	feature_extraction_(consumer_loader, 'Consumer')
	feature_extraction_(shop_loader, 'Shop')
	