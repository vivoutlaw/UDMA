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

import numpy as np
import h5py

from finch import FINCH
offset_s2s_category_ = 100
import pickle
# import ipdb

	
# FINCH Clustering	
def FW_within_catergory(data_, cid_, all_pid_, all_did_):
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
		temp_F_W_pid_ = all_pid_[temp_F_W_idx]
		temp_F_W_did_ = all_did_[temp_F_W_idx]

		feat_ = np.concatenate((temp_F_feat, temp_W_feat))

		c, num_clust, req_c = FINCH(feat_, initial_rank=None, req_clust=None, distance='cosine', verbose=False) #verbose=True
		
		FW_['{}'.format(W_class_[ii_])] = []
		FW_['{}'.format(W_class_[ii_])].append({'idx':temp_F_W_idx, 'c': c, 'nc':num_clust, 'pid_c': temp_F_W_pid_, 'did_c': temp_F_W_did_})
	return FW_


def  data_prep_per_category(all_c_pid_, all_c_cid_, all_s_pid_, all_s_cid_, FW_, finch_layer_, freq_pids, all_s_pid_w_, all_s_feat_):
	map_classes_F_to_W_ = {0:None, 1:107, 2:None, 3:108, 4:None, 5:None, 6:None, 7:None, 8:102, 9:102, 10:109, 11:102, 12:109, 13:110, 14:110, 15:107, 16:110, 17:110, 18:110, 19:110, 20:106, 21:108, 22:None}

	F_FW_W_ = {}

	##############
	#
	# L1:F and L2:FW^{~}
	#
	##############
	list_s2s_idx_ 	= []
	list_s2s_pid_ 	= []
	list_df_pid_  	= []
	list_df_idx_  	= []
	list_sim_ 	= [] 


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
			FW_pid = all_s_pid_[FW_idx]  ################# Here we're using pid's with -1
			FW_pid_c = FW_['{}'.format(temp_W_category_)][0]['pid_c'] ### with all pid's

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

					# Considering all pids obtained within clusters				
					temp_FW_all_pid_ = all_s_pid_[temp_FW_all_pid_idx]
					temp_FW_all_pid_w_ = all_s_pid_w_[temp_FW_all_pid_idx]


					####
					# For weighting
					# Assigning the weak S2S sample to the pid with max number frequency in a given cluster
					####
					temp_FW_diff_pid_ = np.setdiff1d(temp_FW_all_pid_,-1, assume_unique=True) # except -1
					all_unique_pids_ = np.unique(temp_FW_diff_pid_)
					pids_scores_ = []

					for pp_ in all_unique_pids_:
						f_count_ = len(np.where(temp_FW_diff_pid_ == pp_)[0])
						pids_scores_.append(f_count_* freq_pids[pp_])

					pids_scores_ = np.asarray(pids_scores_)

					max_pid_score = max(pids_scores_)
					score_of_current_pid = pids_scores_[np.where(all_unique_pids_==pid_)[0]]

					chosen_df_pids_ = all_unique_pids_[pids_scores_ == max_pid_score]

					# # Obtaining indexes of Fashion pid
					# if len(t_Fs_pid_idx) >1:
					# 	pdb.set_trace()

					# # WS2+
					g_unique_pids_ = chosen_df_pids_
					df_sho_  = torch.zeros((len(g_unique_pids_), 2048))
					df_sho_k = 0
					for pp_ in g_unique_pids_:
						g_df_s_Freq_idx_ = temp_FW_all_pid_idx[np.where(temp_FW_all_pid_ == pp_)[0]]
						g_df_s_Freq_feat = F.normalize(torch.from_numpy(np.mean(all_s_feat_[g_df_s_Freq_idx_],axis=0)).unsqueeze(0), p=2, dim=1)#.numpy()
						df_sho_[df_sho_k,:] = g_df_s_Freq_feat
						df_sho_k +=1

					# Get all S2S pids in this cluster
					# This actually checks if the cluster even had a S2S sample.
					g_s2s_s_idx = np.where(temp_FW_all_pid_ == -1)[0]

					if len(g_s2s_s_idx) >0:
						g_s2s_s_pids = temp_FW_all_pid_w_[g_s2s_s_idx]

						g_s2s_s_pids_real_idx_ = temp_FW_all_pid_idx[g_s2s_s_idx]
						g_s2s_s_feat = F.normalize(torch.from_numpy(all_s_feat_[g_s2s_s_pids_real_idx_]), p=2, dim=1)#.numpy()

						# for g_ii_ in range(g_s2s_s_feat.shape[0]):
						#  	np.append(idx_s2s_df_d_,[[g_s2s_s_pids_real_idx_[g_ii_], g_s2s_s_pids[g_ii_], pid_, mm[g_ii_].item()]], axis=0)

						# if len(g_s2s_s_feat) >1  and len(df_sho_) > 1:
						# 	pdb.set_trace()

						# DF to S2S
						# mm = torch.mm(df_sho_,g_s2s_s_feat.transpose(0,1)).numpy()

						# S2S to DF -- similarity
						mm = torch.mm(g_s2s_s_feat,df_sho_.transpose(0,1)) #.numpy()

						# For each S2S sample, obtaining the most similar DF sample
						# for mm_ in range(mm.shape[0]):
						# 	pdb.set_trace()

						# Maximum similarity S2S sample to mean-DF pid
						similarity_ , pred_ = mm.max(1)

						list_s2s_idx_.append(g_s2s_s_pids_real_idx_.tolist())
						list_s2s_pid_.append(g_s2s_s_pids.tolist())
						if type(g_unique_pids_[pred_]) == np.dtype(np.ndarray):
							list_df_pid_.append(g_unique_pids_[pred_].tolist())
						else:
							list_df_pid_.append(np.asarray([g_unique_pids_[pred_]]).tolist())
						list_sim_.append(similarity_.tolist()) 

						for pp_ in g_unique_pids_:
							g_df_s_Freq_idx_ = temp_FW_all_pid_idx[np.where(temp_FW_all_pid_ == pp_)[0]]
							# Storing indexes of DF for statistics
							list_df_idx_.append(g_df_s_Freq_idx_.tolist())


	list_s2s_idx_ = list(itertools.chain.from_iterable(list_s2s_idx_))
	list_s2s_pid_ = list(itertools.chain.from_iterable(list_s2s_pid_))

	list_df_idx_ = list(itertools.chain.from_iterable(list_df_idx_))
	list_df_pid_ = list(itertools.chain.from_iterable(list_df_pid_))
	
	list_sim_ = list(itertools.chain.from_iterable(list_sim_))

	return list_s2s_idx_, list_s2s_pid_, list_df_idx_, list_df_pid_, list_sim_

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
	data_s_df_feat_ = np.asarray(data_s_df_['X_']) 
	data_s_df_pid_ = np.asarray(data_s_df_['pid']).astype(int)  
	data_s_df_cid_ = np.asarray(data_s_df_['cid']).astype(int)  
	data_s_df_dataset_id_ = np.ones(len(data_s_df_cid_)).astype(int)

	data_s_s2s_ = h5py.File(os.path.join(path_to_data_s2s_,'train_Shop_{}_{}.h5'.format(feat_type_, feat_epoch_num_)),'r')
	data_s_s2s_feat_ = np.asarray(data_s_s2s_['X_']) 
	data_s_s2s_pid_ = np.asarray(data_s_s2s_['pid']).astype(int) + max(data_s_df_pid_)
	data_s_s2s_cid_ = np.asarray(data_s_s2s_['cid']).astype(int) + offset_s2s_category_
	data_s_s2s_dataset_id_ = 2*np.ones(len(data_s_s2s_cid_)).astype(int)
	data_s_s2s_pid_wo_ = -1 * np.ones(len(data_s_s2s_cid_)).astype(int)

	all_s_feat_ = np.concatenate((data_s_df_feat_,data_s_s2s_feat_))
	all_s_pid_wo_ = np.concatenate((data_s_df_pid_,data_s_s2s_pid_wo_))
	all_s_pid_w_ = np.concatenate((data_s_df_pid_,data_s_s2s_pid_))
	all_s_cid_ = np.concatenate((data_s_df_cid_,data_s_s2s_cid_))
	all_s_dataset_id_ = np.concatenate((data_s_df_dataset_id_,data_s_s2s_dataset_id_))

	# For weighting
	pids_ = np.unique(data_s_df_pid_)
	freq_pids = []

	for pp_ in pids_:
		f_count_ = np.where(data_s_df_pid_ == pp_)[0]
		freq_pids.append(1/len(f_count_))
	freq_pids = np.asarray(freq_pids)


	# all_indexes = np.asarray(range(len(all_s_dataset_id_)))

	del data_s_df_, data_s_df_feat_, data_s_df_pid_, data_s_df_cid_, data_s_df_dataset_id_
	del data_s_s2s_, data_s_s2s_feat_, data_s_s2s_pid_, data_s_s2s_cid_, data_s_s2s_dataset_id_ 


	# FW (DF-S2S) within category
	FW_ = FW_within_catergory(all_s_feat_, all_s_cid_, all_s_pid_w_, all_s_dataset_id_)

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

	return FW_, all_s_dataset_id_, all_c_pid_, all_c_cid_, all_s_pid_wo_, all_s_cid_, freq_pids, all_s_pid_w_, all_s_feat_

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


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Expert Matcher')
	parser.add_argument('--finch-part', type=int, default=0)
	args = parser.parse_args()

	#############################
	#
	# Setting default parameters
	#
	#############################

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

	output_finch_pred_path_ = 'features/finch_partitions/{}'.format(model_name)

	if not os.path.isdir(output_finch_pred_path_):
		os.makedirs(output_finch_pred_path_)


	output_finch_pred_filename_pkl = os.path.join(output_finch_pred_path_,'L1_L2_L3_finch_base_X_{}_Finch_{}_List.pkl'.format(feat_epoch_num_,finch_partition_number_))
	output_finch_pred_filename_npz = os.path.join(output_finch_pred_path_,'L1_L2_L3_finch_base_X_{}_Finch_{}_List.npz'.format(feat_epoch_num_,finch_partition_number_))

	if not os.path.isfile(output_finch_pred_filename_npz):
		# Clustering
		FW_, all_s_dataset_id_, all_c_pid_, all_c_cid_, all_s_pid_wo_, all_s_cid_, freq_pids, all_s_pid_w_, all_s_feat_ = cluster_X_base(path_to_data_df_, path_to_data_s2s_, feat_epoch_num_, feat_type_)

		np.savez(output_finch_pred_filename_npz, FW_=FW_, all_s_dataset_id_=all_s_dataset_id_, all_c_pid_=all_c_pid_, all_c_cid_=all_c_cid_,  all_s_pid_wo_=all_s_pid_wo_, 
						all_s_cid_=all_s_cid_, freq_pids=freq_pids, all_s_pid_w_=all_s_pid_w_, all_s_feat_=all_s_feat_)

		# Indexing
		F_FW_W_ = data_prep_per_category(all_c_pid_, all_c_cid_, all_s_pid_wo_, all_s_cid_, FW_, finch_partition_number_, freq_pids, all_s_pid_w_, all_s_feat_)
		with open(output_finch_pred_filename_pkl, 'wb') as f: pickle.dump(F_FW_W_, f, pickle.HIGHEST_PROTOCOL)

	else:
		pred = np.load(output_finch_pred_filename_npz, allow_pickle=True)
		FW_=pred['FW_'].tolist()
		all_s_dataset_id_=pred['all_s_dataset_id_']
		all_c_pid_=pred['all_c_pid_']
		all_c_cid_=pred['all_c_cid_']
		all_s_pid_wo_=pred['all_s_pid_wo_']
		all_s_cid_=pred['all_s_cid_']
		freq_pids = pred['freq_pids']
		all_s_pid_w_ = pred['all_s_pid_w_']
		all_s_feat_ = pred['all_s_feat_']
		# with open(output_finch_pred_filename_pkl, 'rb') as f: F_FW_W_ = pickle.load(f)

		list_s2s_idx_, list_s2s_pid_, list_df_idx_, list_df_pid_, list_sim_ = data_prep_per_category(all_c_pid_, all_c_cid_, all_s_pid_wo_, all_s_cid_, FW_, finch_partition_number_, freq_pids, all_s_pid_w_, all_s_feat_)

		print('S2S. Samples: {},  PID: {} '.format(len(np.unique(np.asarray(list_s2s_idx_))), len(np.unique(np.asarray(list_s2s_pid_))) ))
		print('DF. Samples: {},  PID: {} '.format(len(np.unique(np.asarray(list_df_idx_))), len(np.unique(np.asarray(list_df_pid_)))))

		indexes_array_  = np.asarray(list_s2s_idx_)
		unique_samples_ = np.unique(indexes_array_)

		f = open('{}/L12_WS5_WS6_{}.csv'.format(output_finch_pred_path_,finch_partition_number_),'w')

		for ss_ in unique_samples_:
			temp_idx_ = np.where(indexes_array_ == ss_ )[0]
			for ii_ in range(1):
				# print('Idx: {}, S2S: {}, DF: {}, Sim: {}'.format(list_s2s_idx_[ii_], list_s2s_pid_[ii_], list_df_pid_[ii_], list_sim_[ii_] ))
				f.write('{}, {}, {}, {}\n'.format(list_s2s_idx_[temp_idx_[ii_]], list_s2s_pid_[temp_idx_[ii_]], list_df_pid_[temp_idx_[ii_]], list_sim_[temp_idx_[ii_]] ))
		f.close()



