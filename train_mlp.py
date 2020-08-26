from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import progress_bar, adjust_learning_rate
from datasets import load_Weighting_MLP_data
from model import NAVER_MLP
from torchvision import transforms

import pdb, os
import itertools
import pandas as pd

import torch.nn as nn

from dirtorch.utils import common
import dirtorch.nets as nets

from pytorch_metric_learning import miners, losses

import numpy as np
import h5py

from finch import FINCH
offset_s2s_category_ = 100
import pickle

############################
#
# Training and Testing
#
############################

# Training
def train_mined_net(epoch, train_test_loader, lr):
	# print('\nEpoch: %d' % epoch)
	net.train()

	for batch_idx, (anc_pos_label, target_class) in enumerate(train_test_loader):

		optimizer.zero_grad()

		for ii_ in range(batch_category_size):

			try:
				data = torch.cat((anc_pos_label[target_class[ii_].item()][0]['anc'][0], anc_pos_label[target_class[ii_].item()][0]['pos'][0]), dim=0).to(device)
				labels = torch.cat((anc_pos_label[target_class[ii_].item()][0]['label'][0], anc_pos_label[target_class[ii_].item()][0]['label'][0]), dim=0)
			except:
				pdb.set_trace()

			embeddings = net(data)
			hard_pairs = miner(embeddings, labels)
			loss = loss_func(embeddings, labels, hard_pairs)

			if ii_ == 0:
				total_loss =  loss
			else:
				total_loss = total_loss + loss

		total_loss.backward()
		optimizer.step()

		# Issues with pytorch. Necessary to clear the cache.
		del data, embeddings
		torch.cuda.empty_cache()

		if batch_idx+1 == total_train_step:
			print('Training. Lr {}, Iteration [{}/{}], Loss: {:.6f}'.format(lr, epoch+1, num_epochs, total_loss.item()))
			# print(net.fc.weight)
			# pdb.set_trace()
	
	if np.mod(epoch+1,adjust_learning_rate_step)==0 or np.mod(epoch+1,num_epochs)==0:
		output_dir = '{}/MLP_G_Weighting/{}'.format(checkpoint_path,dataset_name)

		# Save checkpoint.
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)

		print('Saving  Model.')
		if args.optimizer == 'ADAM':
			state = {
				'state_dict': net.state_dict(),
				'optimizer': optimizer.state_dict(),
				'epoch': epoch,
			}
		else:
			state = {
				'state_dict': net.state_dict(),
				'epoch': epoch,
			}

		torch.save(state,'{}_Epoch_{}_ckpt.t7'.format(checkpoint_file_name,epoch))



def load_model(path, iscuda):
	checkpoint = common.load_checkpoint(path, iscuda)
	net = nets.create_model(pretrained="", **checkpoint['model_options'])
	net = common.switch_model_to_cuda(net, iscuda, checkpoint)
	net.load_state_dict(checkpoint['state_dict'])
	net.preprocess = checkpoint.get('preprocess', net.preprocess)
	if 'pca' in checkpoint:
		print('PCA loaded!')
		net.pca = checkpoint.get('pca')
	return net



def without_cluster_X_base():
	########
	#
	# Shop Samples
	#
	########
	data_s_df_ = h5py.File('features/ClusteringFeats/BASE/DeepFashion/DeepFashion_ADAM_ALL/train_Shop_X_60.h5','r')
	data_s_df_feat_ = np.asarray(data_s_df_['X_']) 
	data_s_df_pid_ = np.asarray(data_s_df_['pid']).astype(int)  
	data_s_df_cid_ = np.asarray(data_s_df_['cid']).astype(int)  
	data_s_df_dataset_id_ = np.ones(len(data_s_df_cid_)).astype(int)

	data_s_s2s_ = h5py.File('features/ClusteringFeats/BASE/Street2Shop/DeepFashion_ADAM_ALL/train_Shop_X_60.h5','r')
	data_s_s2s_feat_ = np.asarray(data_s_s2s_['X_']) 
	data_s_s2s_pid_ = np.asarray(data_s_s2s_['pid']).astype(int) + max(data_s_df_pid_)
	data_s_s2s_cid_ = np.asarray(data_s_s2s_['cid']).astype(int) + offset_s2s_category_
	data_s_s2s_dataset_id_ = 2*np.ones(len(data_s_s2s_cid_)).astype(int)

	all_s_feat_ = np.concatenate((data_s_df_feat_,data_s_s2s_feat_))
	all_s_pid_w_ = np.concatenate((data_s_df_pid_,data_s_s2s_pid_))
	all_s_cid_ = np.concatenate((data_s_df_cid_,data_s_s2s_cid_))
	all_s_dataset_id_ = np.concatenate((data_s_df_dataset_id_,data_s_s2s_dataset_id_))

	mask = np.ones(len(all_s_dataset_id_), dtype=bool)

	# Removing classes which has no alignment with DF
	remove_cid_ = np.array([0,1,3,4,5]).astype(int) + offset_s2s_category_

	remove_idx_ = []
	for r_ in remove_cid_:
		remove_idx_.append(np.where(all_s_cid_ == r_)[0].tolist())

	remove_idx_ = list(itertools.chain.from_iterable(remove_idx_))
	mask[remove_idx_] = False
	all_s_feat_ 	= all_s_feat_[mask,:]
	all_s_pid_w_ 	= all_s_pid_w_[mask]
	all_s_cid_ 		= all_s_cid_[mask]
	all_s_dataset_id_ = all_s_dataset_id_[mask]

	# Mean of each pid
	unique_pids_ = np.unique(all_s_pid_w_)
	df_s2s_shop_  = torch.zeros((len(unique_pids_), 2048))
	k_ = 0
	for pp_ in unique_pids_:
		t_idx_ = np.where(all_s_pid_w_ == pp_)[0]
		t_feat_ = F.normalize(torch.from_numpy(np.mean(all_s_feat_[t_idx_],axis=0)).unsqueeze(0), p=2, dim=1)#.numpy()
		df_s2s_shop_[k_,:] = t_feat_
		k_ +=1

	# DF feats and pids
	df_idx_ = np.where(unique_pids_ <= max(data_s_df_pid_))[0]
	df_feats = df_s2s_shop_[df_idx_,:]
	df_pids_ = unique_pids_[df_idx_]

	# S2S feats and pids
	s2s_idx_ = np.where(unique_pids_ > max(data_s_df_pid_))[0]
	s2s_feats = df_s2s_shop_[s2s_idx_,:]
	s2s_pids_ = unique_pids_[s2s_idx_]

	t_df_feats, t_s2s_feats = df_feats.numpy(), s2s_feats.numpy() 
	t_df_pids_, t_s2s_pids_ = df_pids_, s2s_pids_

	# print('DF. Samples: {}, PIDs: {}'.format(len(np.where(all_s_pid_w_ <= max(data_s_df_pid_))[0]), max(data_s_df_pid_)))
	# print('S2S. Samples: {}, PIDs: {}'.format(len(np.where(all_s_pid_w_ > max(data_s_df_pid_))[0]), len(np.unique(s2s_pids_))))

	mm = torch.mm(s2s_feats,df_feats.transpose(0,1))
	sim_, pred_ = mm.max(1)
	sim_, pred_ = sim_.numpy(), pred_.numpy()
	pred_df_pids_ = df_pids_[pred_]

	num_samples_ = sim_.shape[0]

	f = open('WS7.csv','w')

	while num_samples_ is not 0:


		buf_df_pid_ = []
		buf_s2s_pid_ = []

		unique_pred_df_pids_ = np.unique(pred_df_pids_) 
		print('{} {} {} {}'.format(t_df_feats.shape[0], t_s2s_feats.shape[0], t_df_feats.shape[0]-len(unique_pred_df_pids_), num_samples_))

		for uu_ in unique_pred_df_pids_:
			map_ = np.where(pred_df_pids_ == uu_)[0]
			t_idx_ = map_[np.argmax(sim_[map_])]

			t_s2s_pid_ = t_s2s_pids_[t_idx_]
			t_df_pid_ = pred_df_pids_[t_idx_]
			t_sim_ = sim_[t_idx_]

			f.write('{}, {}, {}\n'.format(t_s2s_pid_, t_df_pid_, t_sim_ ))
			buf_s2s_pid_.append(t_s2s_pid_)
			buf_df_pid_.append(t_df_pid_)

		# Remove all the indexes 
		# DF

		t_s2s_pids_  = np.setdiff1d(t_s2s_pids_,np.asarray(buf_s2s_pid_))
		if len(t_s2s_pids_) ==0:
			f.close()
			num_samples_ = 0
			continue
		s_  = []
		for jj_ in t_s2s_pids_:
			s_.append(np.where(s2s_pids_ == jj_)[0].tolist())
		s_ = list(itertools.chain.from_iterable(s_))
		t_s2s_feats = s2s_feats[s_,:]

		t_df_pids_  = np.setdiff1d(t_df_pids_,np.asarray(buf_df_pid_))
		d_  = []
		for jj_ in t_df_pids_:
			d_.append(np.where(df_pids_ == jj_)[0].tolist())
		d_ = list(itertools.chain.from_iterable(d_))
		t_df_feats 	= df_feats[d_,:]

		mm = torch.mm(t_s2s_feats,t_df_feats.transpose(0,1))

		sim_, pred_ = mm.max(1)
		sim_, pred_ = sim_.numpy(), pred_.numpy()
		pred_df_pids_ = t_df_pids_[pred_]

		num_samples_ = sim_.shape[0]





def samples_per_cluster(y_pred):
	u1 = list(set(y_pred))
	size_u1 = np.zeros([len(u1),1])
	map_c_to_ = []
	for k,_ in enumerate(u1):
		ind_ = np.where(y_pred==u1[k])
		size_u1[k,:] = len(ind_[0])
		map_c_to_.append(ind_)
	#LC =max(size_u1)
	#SC = min(size_u1)
	return map_c_to_, size_u1.astype(int) #LC, SC


def shop_base_pid():
	########
	# Shop Samples
	########
	data_s_df_ = h5py.File('features/ClusteringFeats/BASE/DeepFashion/DeepFashion_ADAM_ALL/train_Shop_X_60.h5','r')
	data_s_df_pid_ = np.asarray(data_s_df_['pid']).astype(int)  

	data_s_s2s_ = h5py.File('features/ClusteringFeats/BASE/Street2Shop/DeepFashion_ADAM_ALL/train_Shop_X_60.h5','r')
	data_s_s2s_pid_ = np.asarray(data_s_s2s_['pid']).astype(int) + max(data_s_df_pid_)

	all_s_pid_ = np.concatenate((data_s_df_pid_,data_s_s2s_pid_))

	return all_s_pid_


def combine_with_WS5(F_FW_W_, df_, s2s_idx_, num_classes ):
	new_F_FW_W_ = {}

	for ii_ in range(num_classes):
		new_F_FW_W_[ii_] = []

		for jj_ in range(len(F_FW_W_[ii_])):
			Fc_ = F_FW_W_[ii_][jj_]['Fc']
			Fs_ = F_FW_W_[ii_][jj_]['Fs']
			pid_ = F_FW_W_[ii_][jj_]['pid']

			map_ = np.where(df_ == pid_)[0]

			if len(map_) > 0:
				t_FW_pid_idx = s2s_idx_[map_].tolist()
				new_F_FW_W_[ii_].append({'Fc':Fc_, 'Fs':Fs_, 'FWs': t_FW_pid_idx, 'pid':pid_})

			else:
				new_F_FW_W_[ii_].append({'Fc':Fc_, 'Fs':Fs_, 'pid':pid_})

	return new_F_FW_W_


def combine_with_WS6(F_FW_W_, df_, s2s_pid_, s2s_sim_, all_s_pid_, num_classes ):
	new_F_FW_W_ = {}


	for ii_ in range(num_classes):
		new_F_FW_W_[ii_] = []

		for jj_ in range(len(F_FW_W_[ii_])):
			Fc_ = F_FW_W_[ii_][jj_]['Fc']
			Fs_ = F_FW_W_[ii_][jj_]['Fs']
			pid_ = F_FW_W_[ii_][jj_]['pid']

			map_ = np.where(df_ == pid_)[0]

			if len(map_) > 0:
				t_idx_ = np.argmax(s2s_sim_[map_])

				chosen_pid_ = s2s_pid_[map_[t_idx_]] 
				t_FW_pid_idx = np.where(all_s_pid_ == chosen_pid_)[0].tolist()
				new_F_FW_W_[ii_].append({'Fc':Fc_, 'Fs':Fs_, 'FWs': t_FW_pid_idx, 'pid':pid_})

			else:
				new_F_FW_W_[ii_].append({'Fc':Fc_, 'Fs':Fs_, 'pid':pid_})

	return new_F_FW_W_


def combine_with_WS7(F_FW_W_, df_, s2s_pid_, all_s_pid_, num_classes ):
	new_F_FW_W_ = {}


	for ii_ in range(num_classes):
		new_F_FW_W_[ii_] = []

		for jj_ in range(len(F_FW_W_[ii_])):
			Fc_ = F_FW_W_[ii_][jj_]['Fc']
			Fs_ = F_FW_W_[ii_][jj_]['Fs']
			pid_ = F_FW_W_[ii_][jj_]['pid']

			map_ = np.where(df_ == pid_)[0]

			if len(map_) > 0:
				chosen_pid_ = s2s_pid_[map_]

				t_FW_pid_idx = np.where(all_s_pid_ == chosen_pid_)[0].tolist()
				new_F_FW_W_[ii_].append({'Fc':Fc_, 'Fs':Fs_, 'FWs': t_FW_pid_idx, 'pid':pid_})
			else:
				new_F_FW_W_[ii_].append({'Fc':Fc_, 'Fs':Fs_, 'pid':pid_})

	return new_F_FW_W_




def  final_structure_prepation(list_, F_FW_W_, method_,  num_classes):


	all_s_pid_  = shop_base_pid()

	if method_ == 'WS5' or method_ == 'WS6':
		list_s2s_idx_ 	= list_[0].values
		list_s2s_pid_ 	= list_[1].values
		list_df_pid_ 	= list_[2].values
		list_sim_ 		= list_[3].values

	elif method_ == 'WS7' or method_ == 'WS8':
		list_s2s_pid_ 	= list_[0].values
		list_df_pid_ 	= list_[1].values
		list_sim_ 		= list_[2].values


	if method_ == 'WS5':
		new_F_FW_W_ = combine_with_WS5(F_FW_W_, list_df_pid_, list_s2s_idx_, num_classes )

	elif method_ == 'WS6' or method_ == 'WS8':

		# f = open('WS6_{}.csv'.format(finch_partition_number_),'w')
		# unique_list_s2s_pid_ = np.unique(list_s2s_pid_)
		# for ss_ in unique_list_s2s_pid_:
		# 	temp_idx_ = np.where(list_s2s_pid_ == ss_ )[0]
		# 	if len(temp_idx_) > 1:

		# 		# Maximum similarity
		# 		t_idx_ = np.argmax(list_sim_[temp_idx_])			

		# 		t_s2s_idx_ = list_s2s_idx_[temp_idx_[t_idx_]]
		# 		t_s2s_pid_ = list_s2s_pid_[temp_idx_[t_idx_]]
		# 		t_df_pid_ = list_df_pid_[temp_idx_[t_idx_]]
		# 		t_sim_ = list_sim_[temp_idx_[t_idx_]]
		# 		f.write('{}, {}, {}, {}\n'.format(t_s2s_idx_, t_s2s_pid_,t_df_pid_, t_sim_ ))	
		# 	else:
		# 		f.write('{}, {}, {}, {}\n'.format(list_s2s_idx_[temp_idx_[0]], list_s2s_pid_[temp_idx_[0]], list_df_pid_[temp_idx_[0]], list_sim_[temp_idx_[0]] ))
		# f.close()
		# pdb.set_trace()

		new_F_FW_W_ = combine_with_WS6(F_FW_W_, list_df_pid_, list_s2s_pid_, list_sim_, all_s_pid_, num_classes )

	elif method_ == 'WS7':
		new_F_FW_W_ = combine_with_WS7(F_FW_W_, list_df_pid_, list_s2s_pid_, all_s_pid_, num_classes )

	return new_F_FW_W_

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Expert Matcher')
	parser.add_argument('--batch-size', type=int, default=32, metavar='N',
											help='input batch size for training (default: 128)')
	parser.add_argument('--batch-category-size', type=int, default=8, metavar='N',
											help='input batch size for training (default: 4)')
	parser.add_argument('--num-threads', type=int, default=4, metavar='N',
											help='number of threads')
	parser.add_argument('--epochs', type=int, default=45, metavar='N',
											help='number of epochs to train (default: 10)')
	parser.add_argument('--load-epoch-df', type=int, default=45, metavar='N',
											help='load epoch number')
	parser.add_argument('--load-epoch', type=int, default=45, metavar='N',
											help='load epoch number')
	parser.add_argument('--model', type=str, default='mlp',
											help='mlp')
	parser.add_argument('--optimizer', type=str, default='ADAM',
											help='ADM | SGD')
	parser.add_argument('--comb', type=str, default='L123',
											help='L1 | L12 | L123')
	parser.add_argument('--WS', type=str, default='WS7',
											help='WS5 | WS6 | WS7')
	parser.add_argument('--lr', type=float, default=0.01,
											help='learning rate')
	parser.add_argument('--finch-part', type=int, default=0)
	parser.add_argument('--dataset', type=str, default='DeepFashion',
											help='DeepFashion')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
											help='how many batches to wait before logging training status')
	parser.add_argument('--resume-df', '-rdf', action='store_true')
	parser.add_argument('--resume', '-r', action='store_true')
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
	learning_rate_ = args.lr
	num_epochs = args.epochs
	
	adjust_learning_rate_step  = int(num_epochs/3)


	combinations_type = args.comb # L1, L12, L123
	batch_size = args.batch_size

	dataset_name = args.dataset
	checkpoint_path = 'models'
	finch_partition_number_ = args.finch_part #  0 -> first parition, 1 --> second partition
	batch_category_size = args.batch_category_size
	#############################
	#
	# Generate weak labels for the first time
	#
	#############################

	path_to_BASE_data_ = 'features/ClusteringFeats/BASE'
	model_name = 'DeepFashion_ADAM_ALL'
	feat_epoch_num_ = 60
	feat_type_ = 'X'

	output_finch_pred_path_ = 'features/finch_partitions'
	output_finch_pred_filename_pkl = os.path.join(output_finch_pred_path_,model_name,'L1_L2_L3_finch_base_X_{}_Finch_{}.pkl'.format(feat_epoch_num_,finch_partition_number_))

	with open(output_finch_pred_filename_pkl, 'rb') as f: F_FW_W_ = pickle.load(f)

	weighting_method_ = args.WS
	if weighting_method_ == 'WS5':
		weighting_method_list_ = pd.read_csv('{}/{}/{}_WS5_WS6_{}.csv'.format(output_finch_pred_path_,model_name,combinations_type,finch_partition_number_), header=None)
	elif weighting_method_ == 'WS6':
		# weighting_method_list_ = pd.read_csv('{}/{}/{}_WS5_WS6_{}.csv'.format(output_finch_pred_path_,model_name,combinations_type,finch_partition_number_), header=None)
		weighting_method_list_ = pd.read_csv('{}/{}/{}_WS6_{}.csv'.format(output_finch_pred_path_,model_name,combinations_type,finch_partition_number_), header=None)
	elif weighting_method_ == 'WS7':
		# without_cluster_X_base()
		weighting_method_list_ = pd.read_csv('{}/{}/WS7.csv'.format(output_finch_pred_path_,model_name), header=None)
	elif weighting_method_ == 'WS8':
		weighting_method_list_ = pd.read_csv('{}/{}/WS7.csv'.format(output_finch_pred_path_,model_name), header=None)




	##############################
	#
	# Dataloader and Features
	#
	##############################
	feat_type_2_ = 'X-1'																########################## X or X-1
	path_to_data_df_2_ = os.path.join(path_to_BASE_data_,'DeepFashion',model_name)
	path_to_data_s2s_2_ = os.path.join(path_to_BASE_data_,'Street2Shop',model_name)
	if combinations_type == 'L1' or combinations_type == 'L12':
		num_classes = 28 - 5 # last five classes are samples from S2S 

	# Combining old structure

	F_FW_W_ = final_structure_prepation(weighting_method_list_, F_FW_W_, weighting_method_, num_classes)

	apl_buffer = {}
	trainset =   load_Weighting_MLP_data(path_to_data_df_2_=path_to_data_df_2_, path_to_data_s2s_2_=path_to_data_s2s_2_, feat_epoch_num_=feat_epoch_num_, feat_type_2_=feat_type_2_, 
									num_classes=num_classes, method_type=combinations_type, F_FW_W_=F_FW_W_, batch_size=args.batch_size, apl_buffer=apl_buffer)


	# # Data loader
	trainloader = torch.utils.data.DataLoader(dataset=trainset,# batch_size=batch_category_size, shuffle=True)
											batch_size=batch_category_size,
											shuffle=True,
											# num_workers=args.num_threads, #) #,
											drop_last=True)


	train_test_loader = trainloader
	total_train_step = len(trainloader)


	#############################
	#
	# Model
	#
	#############################

	net = load_model(args.checkpoint, 'cpu')


	#############################
	#
	# Loss and optimizer 
	#
	#############################
	# miner = miners.MultiSimilarityMiner(epsilon=0.1).to(device)
	miner = miners.BatchHardMiner().to(device)
	loss_func = losses.TripletMarginLoss(margin=0.3).to(device)


	#############################
	#
	# Resume 
	#
	#############################

	if dataset_name =='DeepFashion':
		checkpoint_df_model_name = '{}/{}/{}'.format(checkpoint_path,dataset_name,model_name)
		checkpoint_file_name = '{}/MLP_G_Weighting/{}/{}_{}_Finch_{}_{}'.format(checkpoint_path,dataset_name,model_name,combinations_type,finch_partition_number_,weighting_method_)

	if args.resume_df:
		# Load checkpoint.
		load_epoch_num = args.load_epoch_df - 1 
		checkpoint_number = '{}_{}_ckpt.t7'.format(checkpoint_df_model_name,load_epoch_num)
		print(checkpoint_number)
		assert os.path.isdir('models'), 'Error: no checkpoint directory found!'
		checkpoint = torch.load(checkpoint_number)
		net.load_state_dict(checkpoint['state_dict'])
		net =  net.to(device)


	################
	# MLP
	################
	net = NAVER_MLP(naver_model=net).to(device)

	if args.resume:
		# Load checkpoint.
		load_epoch_num = args.load_epoch - 1 
		checkpoint_number = '{}_Epoch_{}_ckpt.t7'.format(checkpoint_file_name,load_epoch_num)
		print(checkpoint_number)
		assert os.path.isdir('models'), 'Error: no checkpoint directory found!'
		checkpoint = torch.load(checkpoint_number)
		net.load_state_dict(checkpoint['state_dict'])
		net =  net.to(device)
		epoch = checkpoint['epoch']
		if args.optimizer == 'adam':
			optimizer.load_state_dict(checkpoint['optimizer'])
	else:
		net =  net.to(device)

	if args.optimizer == 'ADAM':
		optimizer = optim.Adam(net.parameters(), lr=learning_rate_)#1e-3)
	elif args.optimizer == 'SGD':
		optimizer = optim.SGD(net.parameters(), lr=learning_rate_, momentum=0.9, weight_decay=5e-4, nesterov=True)

	if device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True

	#############################
	#
	# Model training ane evaluation 
	#
	#############################
	 
	for epoch in range(0, num_epochs):

		# Adjusting learning rate
		lr = adjust_learning_rate(optimizer, epoch, init_lr=learning_rate_, step=adjust_learning_rate_step, decay=0.1)
		
		train_mined_net(epoch, train_test_loader, lr)


