from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import progress_bar, adjust_learning_rate
from datasets import load_MLP_test_data
from model import NAVER_MLP_Test
from torchvision import transforms
import scipy.io as sio

import pdb, os
import pandas as pd
import h5py

import torch.nn as nn
from dirtorch.utils import common
import dirtorch.nets as nets
import numpy as np




# Testing
def test_DeepFashion_dataset_net(train_test_loader, eval_data_type):
	net.eval()
	with torch.no_grad():

		total_samples = len(train_test_loader.dataset)		
		X_ = torch.zeros([len(train_test_loader.dataset), 2048])
		labels = []
		k=0

		
		for batch_idx, (data, targets) in enumerate(train_test_loader):

			print('[{}/{}]'.format(batch_idx,len(train_test_loader)))	
			
			# Data and labels
			data = data.to(device)

			# Text data
			targets = np.asarray(targets[0])

			# NAVER Triplet Network
			# x_ = net.get_embedding(data)
			
			x_ = net(data)

			# Trained features
			X_[k*batch_size:(k*batch_size)+data.shape[0],:] = x_.view(data.shape[0],x_.shape[1])
			labels = np.concatenate((labels,targets))
			k+=1

		X_ 	= X_.cpu().detach().numpy()

		output_dir = 'features/{}'.format(features_output_file_path)
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)


		sio.savemat('{}/{}_X-1_{}.mat'.format(output_dir,eval_data_type,load_epoch_num+1),{'feats':X_})
		
		labels = pd.DataFrame(labels)
		labels.to_csv('{}/{}_Labels_{}.txt'.format(output_dir,eval_data_type,load_epoch_num+1), index = None, header=None)


def test_Street2Shop_dataset_net(train_test_loader, eval_data_type):
	net.eval()
	with torch.no_grad():

		total_samples = len(train_test_loader.dataset)		
		X_ = torch.zeros([len(train_test_loader.dataset), 2048])
		k=0
		
		for batch_idx, (data) in enumerate(train_test_loader):

			print('[{}/{}]'.format(batch_idx,len(train_test_loader)))	
			
			# Data and labels
			data = data.to(device)

			x_ = net(data)

			# Trained features
			X_[k*batch_size:(k*batch_size)+data.shape[0],:] = x_.view(data.shape[0],x_.shape[1])
			k+=1

		X_ 	= X_.cpu().detach().numpy()

		output_dir = '{}'.format(features_output_file_path)
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)

		with h5py.File('{}/{}_X_{}_Crop.h5'.format(output_dir,eval_data_type,load_epoch_num+1), 'w') as hf:
			hf.create_dataset('X_', data=X_, dtype='float32')


		
		

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
	parser.add_argument('--comb', type=str, default='L123',
											help='L1 | L12 | L123')
	parser.add_argument('--model-name', type=str, default='DeepFashion',
											help='DeepFashion | DF_S2S ')
	parser.add_argument('--eval-dataset', type=str, default='XXX',
											help='DeepFashion | Street2Shop ')
	parser.add_argument('--seed', type=int, default=100, metavar='S',
											help='random seed (default: 1)')
	parser.add_argument('--WS', type=str, default=None,
											help='WS5 | WS6 | WS7')
	parser.add_argument('--resume', '-r', action='store_true')
	parser.add_argument('--finch-part', type=int, default=0)
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

	# Seeding
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	if device == 'cuda':
		torch.cuda.manual_seed(args.seed)

	batch_size = args.batch_size
	model_name = args.model_name

	combinations_type = args.comb # L1, L12, L123

	checkpoint_path = 'models'
	train_test_type = 'test' 
	eval_dataset_name = args.eval_dataset
	finch_partition_number_ = args.finch_part #  0 -> first parition, 1 --> second partition


	#############################
	#
	# Paths
	#
	#############################
	path_to_BASE_data_ = 'features'
	base_model_name = 'DeepFashion_ADAM_ALL'
	feat_epoch_num_ = 60
	feat_type_ = 'X-1'																########################## X or X-1
	# feat_type_ = 'X'
	path_to_data_df_ = os.path.join(path_to_BASE_data_,'DeepFashion',base_model_name)
	path_to_data_s2s_ = os.path.join(path_to_BASE_data_,'Street2Shop',base_model_name)


	if eval_dataset_name == 'DeepFashion':
		path_to_consumer_ = '{}/Query_{}_{}.mat'.format(path_to_data_df_,feat_type_,feat_epoch_num_)
		path_to_shop_ = '{}/Gallery_{}_{}.mat'.format(path_to_data_df_,feat_type_,feat_epoch_num_)

	elif eval_dataset_name == 'Street2Shop':
		if feat_type_ == 'X-1':
			path_to_consumer_ = '{}/Query_{}_{}_Crop.h5'.format(path_to_data_s2s_,feat_type_,feat_epoch_num_)
		else:
			path_to_consumer_ = '{}/Query_{}_{}_Crop.h5'.format(path_to_data_s2s_,feat_type_,feat_epoch_num_)

		path_to_shop_ = '{}/Gallery_{}_{}_Crop.h5'.format(path_to_data_s2s_,feat_type_,feat_epoch_num_)


	#############################
	#
	# Dataloader
	#
	#############################

	if eval_dataset_name =='DeepFashion':
		print('No DF eval yet')

	elif eval_dataset_name == 'Street2Shop':
		
		consumer_set = load_MLP_test_data(path_to_data_=path_to_consumer_)
		shop_set = load_MLP_test_data(path_to_data_=path_to_shop_)



	# Data loader
	consumer_loader = torch.utils.data.DataLoader(dataset=consumer_set,#batch_size=10)
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

	net = NAVER_MLP_Test()

	######################
	# For loading models
	#
	#####################
	if model_name =='DeepFashion':
		if args.WS is None:
			checkpoint_file_name = '{}/MLP_G_Weighting/{}/{}_{}_Finch_{}'.format(checkpoint_path,model_name,base_model_name,combinations_type,finch_partition_number_)
			features_output_file_path = 'features/MLP_G_Weighting/{}/{}_{}_{}'.format(eval_dataset_name,base_model_name,combinations_type,finch_partition_number_)
		else:
			checkpoint_file_name = '{}/MLP_G_Weighting/{}/{}_{}_Finch_{}_{}'.format(checkpoint_path,model_name,base_model_name,combinations_type,finch_partition_number_,args.WS)
			features_output_file_path = 'features/MLP_G_Weighting/{}/{}_{}_{}_{}'.format(eval_dataset_name,base_model_name,combinations_type,finch_partition_number_,args.WS)


	######################
	# For storing features
	#
	######################

	load_epoch_num = args.load_epoch - 1 

	if args.resume:
		# Load checkpoint.
		checkpoint_number = '{}_Epoch_{}_ckpt.t7'.format(checkpoint_file_name,load_epoch_num)
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

	if eval_dataset_name =='DeepFashion':
	 
		test_DeepFashion_dataset_net(consumer_loader, 'Query')
		test_DeepFashion_dataset_net(shop_loader, 'Gallery')

	elif eval_dataset_name == 'Street2Shop':

		test_Street2Shop_dataset_net(consumer_loader, 'Query')
		test_Street2Shop_dataset_net(shop_loader, 'Gallery')
		
