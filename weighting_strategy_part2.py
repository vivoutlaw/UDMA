from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision import transforms
import pdb, os
import itertools
import pandas as pd
import torch.nn as nn
from dirtorch.utils import common
import dirtorch.nets as nets
import numpy as np
import h5py
from finch import FINCH
offset_s2s_category_ = 100
import pickle



# FINCH Clustering
def W_within_catergory(data_, cid_):
	W_clustering_within_category = np.asarray([0, 1, 3, 4, 5]) + offset_s2s_category_ ;

	W_ = {}
	for ii_ in W_clustering_within_category:
		temp_idx = np.where(cid_ == ii_)[0]
		feat_ = data_[temp_idx, :]
		c, num_clust, req_c = FINCH(feat_, initial_rank=None, req_clust=None, distance='cosine', verbose=False) #verbose=True
		
		W_['{}'.format(ii_)] = []
		W_['{}'.format(ii_)].append({'idx':temp_idx, 'c': c, 'nc':num_clust})
	return W_
		
def FW_within_catergory(data_, cid_):
	W_class_ = np.asarray([2, 6, 7, 8, 9, 10]) + offset_s2s_category_
	FW_clustering_within_category_F_ = {0:[8, 9, 11], 1:[20], 2:[1,15], 3:[3,21], 4:[10,12], 5:[13, 14, 16, 17, 18, 19]}

	FW_ = {}

	for ii_ in range(len(W_class_)):

		# S2S
		temp_W_category = W_class_[ii_]
		temp_W_idx = np.where(cid_ == temp_W_category)[0]
		temp_W_feat = data_[temp_W_idx,:]

		# DF
		temp_F_categories = FW_clustering_within_category_F_[ii_]
		temp_F_idx = []
		for jj_ in temp_F_categories:
			temp_idx_ = np.where(cid_ ==  jj_)[0]
			temp_F_idx = np.concatenate((temp_F_idx,temp_idx_)).astype(int)
		
		temp_F_feat = data_[temp_F_idx,:]

		temp_F_W_idx = np.concatenate((temp_F_idx, temp_W_idx)).astype(int)
		feat_ = np.concatenate((temp_F_feat, temp_W_feat))

		c, num_clust, req_c = FINCH(feat_, initial_rank=None, req_clust=None, distance='cosine', verbose=False) #verbose=True
		
		FW_['{}'.format(W_class_[ii_])] = []
		FW_['{}'.format(W_class_[ii_])].append({'idx':temp_F_W_idx, 'c': c, 'nc':num_clust})
	return FW_


def  data_prep_per_category(all_c_pid_, all_c_cid_, all_s_pid_, all_s_cid_, FW_, W_, finch_layer_):
	map_classes_F_to_W_ = {0:None, 1:107, 2:None, 3:108, 4:None, 5:None, 6:None, 7:None, 8:102, 9:102, 10:109, 11:102, 12:109, 13:110, 14:110, 15:107, 16:110, 17:110, 18:110, 19:110, 20:106, 21:108, 22:None}

	add_new_classes_to_W_to_F_ = {100:23, 101:24, 103:25, 104:26, 105:27}

	F_FW_W_ = {}

	##############
	#
	# L3:W^{~}
	#
	##############
	offset_pid_for_W_L3_ = len(np.unique(all_c_pid_)) + offset_s2s_category_
	k_ = offset_pid_for_W_L3_

	for W_cat_ in add_new_classes_to_W_to_F_:

		# L3:W^{~}
		F_FW_W_[add_new_classes_to_W_to_F_[W_cat_]] = []
		temp_W_category_ = W_cat_

		W_idx = W_['{}'.format(temp_W_category_)][0]['idx']
		W_clust_id = W_['{}'.format(temp_W_category_)][0]['c'][:,finch_layer_]

		for t_idx, clid_ in enumerate(W_clust_id):
			# Shop 
			t_W_clid_ = np.where(W_clust_id == clid_)[0]
			t_W_clid_idx_ = W_idx[t_W_clid_]

			F_FW_W_[add_new_classes_to_W_to_F_[W_cat_]].append({'Ws':t_W_clid_idx_.tolist(), 'pid':k_})
			k_ = k_ + 1

	##############
	#
	# L1:F and L2:FW^{~}
	#
	##############
	for F_cat_ in map_classes_F_to_W_:
		# print(F_cat_)

		F_FW_W_[F_cat_] = []

		temp_W_category_ = map_classes_F_to_W_[F_cat_]

		temp_F_category_idx_ = np.where(all_c_cid_ == F_cat_)[0]
		temp_F_pids_ = all_c_pid_[temp_F_category_idx_]
		temp_F_pids_unique_ = np.unique(temp_F_pids_)

		if temp_W_category_ is not None:

			# L2 :  FW^{~}
			FW_idx = FW_['{}'.format(temp_W_category_)][0]['idx']
			FW_clust_id = FW_['{}'.format(temp_W_category_)][0]['c'][:,finch_layer_]
			FW_pid = all_s_pid_[FW_idx]

			for pid_ in temp_F_pids_unique_:
				# Consumer
				t_F_pid_idx = np.where(all_c_pid_ == pid_)[0]

				# Shop 
				t_Fs_pid_idx = np.where(all_s_pid_ == pid_)[0]

				# Shop: FW^{~}
				temp_FW_pid_idx = np.where(FW_pid == pid_)[0]
				temp_cluster_number_ = np.unique(FW_clust_id[temp_FW_pid_idx])

				# if len(temp_cluster_number_) >1:
				# 	pdb.set_trace()

				t_FW_pid_idx = []

				for cc_ in temp_cluster_number_:
					temp_cluster_idx_ = np.where(FW_clust_id == cc_)[0]
					temp_FW_all_pid_idx = FW_idx[temp_cluster_idx_]

					# Only indexes obtained via clusters					
					temp_FW_diff_idx_ = np.setdiff1d(temp_FW_all_pid_idx, t_Fs_pid_idx)
					temp_FW_diff_pid_ = all_s_pid_[temp_FW_diff_idx_]

					# Excluding all other produdct ids from Fashion dataset, and only considering the S2S samples
					final_FW_diff_idx_ = temp_FW_diff_idx_[temp_FW_diff_pid_ == -1]

					if len(final_FW_diff_idx_) > 0:
						t_FW_pid_idx.append(final_FW_diff_idx_.tolist())

				t_FW_pid_idx = list(itertools.chain.from_iterable(t_FW_pid_idx))

				if len(t_FW_pid_idx):
					F_FW_W_[F_cat_].append({'Fc':t_F_pid_idx.tolist(), 'Fs':t_Fs_pid_idx.tolist(), 'FWs': t_FW_pid_idx, 'pid':pid_})				
				else: 
					F_FW_W_[F_cat_].append({'Fc':t_F_pid_idx.tolist(), 'Fs':t_Fs_pid_idx.tolist(), 'pid':pid_})

		else:
			for pid_ in temp_F_pids_unique_:
				# Consumer
				t_F_pid_idx = np.where(all_c_pid_ == pid_)[0]

				# Shop 
				t_Fs_pid_idx = np.where(all_s_pid_ == pid_)[0]

				F_FW_W_[F_cat_].append({'Fc':t_F_pid_idx.tolist(), 'Fs':t_Fs_pid_idx.tolist(), 'pid':pid_})

	return F_FW_W_

def  data_prep_all_category(all_c_pid_, all_s_pid_):

	Fc_Fs_ = []

	temp_F_pids_unique_ = np.unique(all_c_pid_)
	
	for pid_ in temp_F_pids_unique_:
		# Consumer
		t_Fc_pid_idx = np.where(all_c_pid_ == pid_)[0]

		# Shop 
		t_Fs_pid_idx = np.where(all_s_pid_ == pid_)[0]

		Fc_Fs_.append({'Fc':t_Fc_pid_idx.tolist(), 'Fs':t_Fs_pid_idx.tolist(), 'pid':pid_})

	return Fc_Fs_


def cluster_X_base(path_to_data_df_, path_to_data_s2s_, feat_epoch_num_, feat_type_):
	########
	#
	# Shop Samples
	#
	########
	data_s_df_ = h5py.File(os.path.join(path_to_data_df_,'train_Shop_{}_{}.h5'.format(feat_type_, feat_epoch_num_)),'r')
	data_s_df_feat_ = np.asarray(data_s_df_['X_']) #-- BEFORE

	# # TO COMMENT -- AFTER
	# feat_path =  h5py.File('/cvhci/data/fashion_NAVER/NAVER_Codes/deep-image-retrieval/kntorch/features/ClusteringFeats/CURR/DeepFashion/DeepFashion_ADAM_ALL_L12_0/train_Shop_X+1_0.h5','r')
	# data_s_df_feat_ = np.asarray(feat_path['X_']) # -- AFTER

	data_s_df_pid_ = np.asarray(data_s_df_['pid']).astype(int)  
	data_s_df_cid_ = np.asarray(data_s_df_['cid']).astype(int)  
	data_s_df_dataset_id_ = np.ones(len(data_s_df_cid_)).astype(int)

	data_s_s2s_ = h5py.File(os.path.join(path_to_data_s2s_,'train_Shop_{}_{}.h5'.format(feat_type_, feat_epoch_num_)),'r')
	data_s_s2s_feat_ = np.asarray(data_s_s2s_['X_']) #-- BEFORE

	# # TO COMMENT -- AFTER
	# feat_path =  h5py.File('/cvhci/data/fashion_NAVER/NAVER_Codes/deep-image-retrieval/kntorch/features/ClusteringFeats/CURR/Street2Shop/DeepFashion_ADAM_ALL_L12_0/train_Shop_X+1_0.h5','r')
	# data_s_s2s_feat_ = np.asarray(feat_path['X_']) # -- AFTER

	# data_s_s2s_pid_ = np.asarray(data_s_s2s_['pid']).astype(int)  
	data_s_s2s_cid_ = np.asarray(data_s_s2s_['cid']).astype(int) + offset_s2s_category_
	data_s_s2s_dataset_id_ = 2*np.ones(len(data_s_s2s_cid_)).astype(int)
	data_s_s2s_pid_ = -1 * np.ones(len(data_s_s2s_cid_)).astype(int)

	all_s_feat_ = np.concatenate((data_s_df_feat_,data_s_s2s_feat_))
	all_s_pid_ = np.concatenate((data_s_df_pid_,data_s_s2s_pid_))
	all_s_cid_ = np.concatenate((data_s_df_cid_,data_s_s2s_cid_))
	all_s_dataset_id_ = np.concatenate((data_s_df_dataset_id_,data_s_s2s_dataset_id_))

	# all_indexes = np.asarray(range(len(all_s_dataset_id_)))

	del data_s_df_, data_s_df_feat_, data_s_df_pid_, data_s_df_cid_, data_s_df_dataset_id_
	del data_s_s2s_, data_s_s2s_feat_, data_s_s2s_pid_, data_s_s2s_cid_, data_s_s2s_dataset_id_ 

	# S2S within category
	W_ = W_within_catergory(all_s_feat_, all_s_cid_)

	# FW (DF-S2S) within category
	FW_ = FW_within_catergory(all_s_feat_, all_s_cid_)

	########
	#
	# Consumer Samples
	#
	########
	all_c_df_ = h5py.File(os.path.join(path_to_data_df_,'train_Consumer_X_{}.h5'.format(feat_epoch_num_)),'r')
	# data_c_df_feat_ = np.asarray(all_c_df_['X_']) 
	all_c_pid_ = np.asarray(all_c_df_['pid']).astype(int)  
	all_c_cid_ = np.asarray(all_c_df_['cid']).astype(int)
	all_c_idx_ = np.asarray(range(len(all_c_cid_)))

	return W_, FW_, all_s_dataset_id_, all_c_pid_, all_c_cid_, all_s_pid_, all_s_cid_




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Expert Matcher')
	parser.add_argument('--optimizer', type=str, default='ADAM',
											help='ADM | SGD')
	parser.add_argument('--comb', type=str, default='L123',
											help='L1 | L12 | L123')
	parser.add_argument('--finch-part', type=int, default=0)
	args = parser.parse_args()

	#############################
	#
	# Setting default parameters
	#
	#############################

	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	combinations_type = args.comb # L1, L12, L123
	finch_partition_number_ = args.finch_part #  0 -> first parition, 1 --> second partition
	#############################
	#
	# Generate weak labels for the first time
	#
	#############################

	path_to_BASE_data_ = 'features/ClusteringFeats/BASE'
	model_name = 'DeepFashion_ADAM_ALL'
	feat_epoch_num_ = 60
	feat_type_ = 'X'
	path_to_data_df_ = os.path.join(path_to_BASE_data_,'DeepFashion',model_name)
	path_to_data_s2s_ = os.path.join(path_to_BASE_data_,'Street2Shop',model_name)

	output_finch_pred_path_ = 'features/finch_partitions'
	output_finch_pred_filename_npz = os.path.join(output_finch_pred_path_,model_name,'L1_L2_L3_finch_base_X_{}_Finch_{}.npz'.format(feat_epoch_num_,finch_partition_number_))
	output_finch_pred_filename_pkl = os.path.join(output_finch_pred_path_,model_name,'L1_L2_L3_finch_base_X_{}_Finch_{}.pkl'.format(feat_epoch_num_,finch_partition_number_))

	if not os.path.isfile(output_finch_pred_filename_pkl):
		#	# Clustering
		W_, FW_, all_s_dataset_id_, all_c_pid_, all_c_cid_, all_s_pid_, all_s_cid_ = cluster_X_base(path_to_data_df_, path_to_data_s2s_, feat_epoch_num_, feat_type_)

		#	# Indexing
		F_FW_W_ = data_prep_per_category(all_c_pid_, all_c_cid_, all_s_pid_, all_s_cid_, FW_, W_, finch_partition_number_)
		Fc_Fs   = data_prep_all_category(all_c_pid_, all_s_pid_)
		np.savez(output_finch_pred_filename_npz, W_=W_, FW_=FW_, all_s_dataset_id_=all_s_dataset_id_, all_c_pid_=all_c_pid_, all_c_cid_=all_c_cid_,  all_s_pid_=all_s_pid_, all_s_cid_=all_s_cid_, Fc_Fs=Fc_Fs)
		with open(output_finch_pred_filename_pkl, 'wb') as f: pickle.dump(F_FW_W_, f, pickle.HIGHEST_PROTOCOL)
