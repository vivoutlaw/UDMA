# -*- coding: utf-8 -*-

from __future__ import print_function
import json

import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import h5py
from torch.nn import functional as F

import io
import pdb
import glob

import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import scipy.misc
import random
import itertools



def default_image_loader(path): # image_size
	img = Image.open(path).convert('RGB')
	# img = img.resize([image_size, image_size])
	return img



def default_loader(feats, label):

		H = torch.from_numpy(feats)
		L = torch.from_numpy(np.array(int(label)))

		return H, L

def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg , (1, 2, 0)))
	plt.show()

def  imshow_new(img_stack):
	fig=plt.figure(figsize=(3, 1))
	columns = 3
	rows = 1
	for i in range(1, columns*rows +1):
		img = img_stack[i-1]
		fig.add_subplot(rows, columns, i)
		plt.imshow(img)
	plt.show()

def visualize_triplet(abs_path, anchor_path,pos_path,neg_path, output_size_):
	a = os.path.join(abs_path,anchor_path)
	b = os.path.join(abs_path,pos_path)
	c = os.path.join(abs_path,neg_path)
	print(a)
	print(b)
	print(c)

	a_, b_, c_ = scipy.misc.imread(a), scipy.misc.imread(b), scipy.misc.imread(c)
	a_, b_, c_ = scipy.misc.imresize(a_,[output_size_,output_size_]), scipy.misc.imresize(b_,[output_size_,output_size_]), scipy.misc.imresize(c_,[output_size_,output_size_])
	abc = np.stack([a_, b_, c_])
	imshow_new(abc)

def set2diff(list1, list2):
	diff_list = list(set(list1).difference(set(list2)))
	return diff_list

def get_path_to_images(id_to_img_path, id):

	temp_num_of_samples = int(id_to_img_path[id][0]['num_samples'])
	temp_chosen_idx = np.random.randint(temp_num_of_samples, size=1)[0]
	temp_chosen_id = id_to_img_path[id][0]['path'][temp_chosen_idx]

	return temp_chosen_id


def  triplet_paths(abs_path, anchor_path,pos_path,neg_path):
	a = os.path.join(abs_path,anchor_path)
	b = os.path.join(abs_path,pos_path)
	c = os.path.join(abs_path,neg_path)

	return a, b, c


def neg_c2s_pos_neg_c2c_s2s(path_to_images_, json_hash_id, json_hash_id_images, 
							anchor_img_path_consumer_, anchor_consumer_dataset, anchor_consumer_id, 
							positive_img_path_shop_, positive_shop_dataset, positive_shop_id, 
							combinations_type, output_size_):
	
	#################
	#
	# Consumer to Shop | Consumer to Consumer
	#
	#################

	temp_dataset_name_ = anchor_consumer_dataset
	temp_consumer_anchor_id = [anchor_consumer_id]


	#################
	#
	# Shop to Shop
	#
	#################
	temp_dataset_name__ = positive_shop_dataset
	temp_shop_positive_id = [positive_shop_id]

	a_ = []
	p_ = []
	n_ = []
	
	if combinations_type == 'C2S':

		#################################
		# [C2S]
		# 
		# Negative pairs for consumer to shop
		# Not considering samples from the same ID
		#
		
		hash_dataset_shop_ids = json_hash_id['All']['shop'][temp_dataset_name_][0]
		neg_ids_c2s = set2diff(hash_dataset_shop_ids,temp_consumer_anchor_id)

		chosen_c2s_idx = np.random.randint(len(neg_ids_c2s), size=1)[0]
		chosen_c2s_id = neg_ids_c2s[chosen_c2s_idx]

		hash_id_to_images_c2s =  json_hash_id_images['All']['shop'][temp_dataset_name_] 
		neg_c2s_img_name = get_path_to_images(hash_id_to_images_c2s, chosen_c2s_id)
		neg_c2s_img_path = os.path.join(temp_dataset_name_,chosen_c2s_id,neg_c2s_img_name)

		# VISUALISE
		# visualize_triplet(path_to_images_, anchor_img_path_consumer_, positive_img_path_shop_, neg_c2s_img_path, output_size_)
		a_, p_, n_ = triplet_paths(path_to_images_, anchor_img_path_consumer_, positive_img_path_shop_, neg_c2s_img_path)

	elif combinations_type == 'ALL': 

		#################################
		# [C2S]
		# 
		# Negative pairs for consumer to shop
		# Not considering samples from the same ID
		#
		
		hash_dataset_shop_ids = json_hash_id['All']['shop'][temp_dataset_name_][0]
		neg_ids_c2s = set2diff(hash_dataset_shop_ids,temp_consumer_anchor_id)

		chosen_c2s_idx = np.random.randint(len(neg_ids_c2s), size=1)[0]
		chosen_c2s_id = neg_ids_c2s[chosen_c2s_idx]

		hash_id_to_images_c2s =  json_hash_id_images['All']['shop'][temp_dataset_name_] 
		neg_c2s_img_name = get_path_to_images(hash_id_to_images_c2s, chosen_c2s_id)
		neg_c2s_img_path = os.path.join(temp_dataset_name_,chosen_c2s_id,neg_c2s_img_name)

		# VISUALISE
		# visualize_triplet(path_to_images_, anchor_img_path_consumer_, positive_img_path_shop_, neg_c2s_img_path, output_size_)
		a1, p1, n1 = triplet_paths(path_to_images_, anchor_img_path_consumer_, positive_img_path_shop_, neg_c2s_img_path)
		

		#################
		#
		# Consumer to Consumer
		#
		#################
		##################################
		# [C2C]
		# 
		# If want to make negatives of consumer to consumer
		#

		hash_dataset_consumer_ids = json_hash_id['All']['consumer'][temp_dataset_name_][0]
		neg_c2c_ids = set2diff(hash_dataset_consumer_ids,temp_consumer_anchor_id)

		chosen_c2c_idx = np.random.randint(len(neg_c2c_ids), size=1)[0]
		chosen_c2c_id = neg_c2c_ids[chosen_c2c_idx]

		hash_id_to_images_c2c = json_hash_id_images['All']['consumer'][temp_dataset_name_] 
		neg_c2c_img_name = get_path_to_images(hash_id_to_images_c2c, chosen_c2c_id)
		neg_c2c_img_path = os.path.join(temp_dataset_name_,chosen_c2c_id,neg_c2c_img_name)

		# Get a positive pair for the consumer
		# get_path_to_images
		pos_c2c_img_name = get_path_to_images(hash_id_to_images_c2c, anchor_consumer_id)
		pos_c2c_img_path = os.path.join(temp_dataset_name_,anchor_consumer_id,pos_c2c_img_name)

		# VISUALISE
		# visualize_triplet(path_to_images_, anchor_img_path_consumer_, pos_c2c_img_path, neg_c2c_img_path, output_size_)
		a2, p2, n2 = triplet_paths(path_to_images_, anchor_img_path_consumer_, pos_c2c_img_path, neg_c2c_img_path)

		###################################
		# [S2S]
		#
		# Negative pairs for shop to shop
		#
		hash_dataset_shop_ids = json_hash_id['All']['shop'][temp_dataset_name__][0]
		neg_ids_s2s = set2diff(hash_dataset_shop_ids,temp_shop_positive_id)

		chosen_s2s_idx = np.random.randint(len(neg_ids_s2s), size=1)[0]
		chosen_s2s_id = neg_ids_s2s[chosen_s2s_idx]

		hash_id_to_images_s2s = json_hash_id_images['All']['shop'][temp_dataset_name__] 
		neg_s2s_img_name = get_path_to_images(hash_id_to_images_s2s, chosen_s2s_id)
		neg_s2s_img_path = os.path.join(temp_dataset_name__,chosen_s2s_id,neg_s2s_img_name)

		# Get a positive pair for the shop
		# get_path_to_images
		pos_s2s_img_name = get_path_to_images(hash_id_to_images_s2s, positive_shop_id)
		pos_s2s_img_path = os.path.join(temp_dataset_name__,positive_shop_id,pos_s2s_img_name)

		# VISUALISE
		# visualize_triplet(path_to_images_, positive_img_path_shop_, pos_s2s_img_path, neg_s2s_img_path, output_size_)
		a3, p3, n3 = triplet_paths(path_to_images_, positive_img_path_shop_, pos_s2s_img_path, neg_s2s_img_path)

		a_ = [a1, a2, a3]
		p_ = [p1, p2, p3]
		n_ = [n1, n2, n3]

	return a_, p_, n_



class load_dataset(Dataset):
		def __init__(self,  path_to_data='XXXX', train_test_type='XXXX', dataset_name='XXXX', path_to_hash='XXXX', path_to_hash_to_images_id='XXXX', path_to_images_='XXXX', 
			output_size_='XXXX', combinations_type='XXXX',transform1=None, transform2=None, transform3=None, loader=neg_c2s_pos_neg_c2c_s2s, image_loader= default_image_loader):

			self.dataset_name = dataset_name

			if self.dataset_name == 'DeepFashion':

				imagelist = pd.read_csv(path_to_data, delimiter=' ', names=['image_pair_name_1', 'image_pair_name_2', 'item_id', 'evaluation_status'])

				if train_test_type =='train':
					sample_set = imagelist[imagelist['evaluation_status']=='train']
				elif train_test_type =='trainval':
					test_ids = imagelist[imagelist['evaluation_status']=='test'].index
					sample_set = imagelist.drop(test_ids)

				sample_set = sample_set.reset_index()


				#####
				#    ANCHOR
				####

				# Obtaining anchor and positive pair
				anchor_img_path_consumer_ = sample_set['image_pair_name_1']
				anchor_consumer_dataset = pd.Series(anchor_img_path_consumer_).str.split(pat = "/id_", expand=True)[0]
				# anchor id is obtained from the anchor path, in order to avoid any confusion.
				anchor_consumer_id = pd.Series(anchor_img_path_consumer_).str.split(pat = "/", expand=True)[2]
				# anchor_id = sample_set['item_id']

				self.anchor_img_path_consumer_ = anchor_img_path_consumer_
				self.anchor_consumer_dataset = anchor_consumer_dataset
				self.anchor_consumer_id = anchor_consumer_id

				#####
				#    POSITIVES
				####

				positive_img_path_shop_ = sample_set['image_pair_name_2']
				positive_shop_dataset = pd.Series(positive_img_path_shop_).str.split(pat = "/id_", expand=True)[0]

				# anchor id is obtained from the anchor path, in order to avoid any confusion.
				positive_shop_id = pd.Series(positive_img_path_shop_).str.split(pat = "/", expand=True)[2]
				# anchor_id = sample_set['item_id']

				self.positive_img_path_shop_ = positive_img_path_shop_
				self.positive_shop_dataset = positive_shop_dataset
				self.positive_shop_id = positive_shop_id


				#####  NEGATIVES
				#
				#   Making pairs for (1) consumer to shop (C2S).
				#       - For future, considesr making pairs for (2) consumser to consumer (C2C) and (3) shop to shop (S2S)
				#
				####
				# Obtaining negative id for a given anchor
				self.json_hash_id = json.load(open(path_to_hash))
				self.json_hash_id_images = json.load(open(path_to_hash_to_images_id))

				self.loader = neg_c2s_pos_neg_c2c_s2s
				self.path_to_images_ = path_to_images_
				self.output_size_ = output_size_
				self.combinations_type = combinations_type
				self.transform1 = transform1
				self.transform2 = transform2
				self.transform3 = transform3
				self.image_loader = default_image_loader


		def __getitem__(self, index):

			if self.dataset_name == 'DeepFashion':
				anchor_paths, pos_paths, neg_paths = self.loader(self.path_to_images_, self.json_hash_id, self.json_hash_id_images, 
												self.anchor_img_path_consumer_[index], self.anchor_consumer_dataset[index], self.anchor_consumer_id[index],
												self.positive_img_path_shop_[index], self.positive_shop_dataset[index], self.positive_shop_id[index], 
												self.combinations_type, self.output_size_)

				if self.combinations_type == 'C2S':
					anchor_data = self.transform1(self.image_loader(anchor_paths))
					pos_data 	= self.transform1(self.image_loader(pos_paths))
					neg_data 	= self.transform1(self.image_loader(neg_paths))

				elif self.combinations_type == 'ALL':
					anchor_data = torch.stack(list(self.transform1(self.image_loader(pth_)) for pth_ in anchor_paths))
					pos_data 	= torch.stack(list(self.transform2(self.image_loader(pth_)) for pth_ in pos_paths))
					neg_data 	= torch.stack(list(self.transform3(self.image_loader(pth_)) for pth_ in neg_paths))

			return anchor_data, pos_data, neg_data

		def __len__(self):
			return len(self.anchor_img_path_consumer_)


class load_test_df_dataset(Dataset):
		def __init__(self,  path_to_images_='XXXX', path_to_txt_='XXXX', dataset_name='XXXX',transform=None, image_loader= default_image_loader):

			self.dataset_name = dataset_name

			if self.dataset_name == 'DeepFashion':
				imagelist = pd.read_csv(open(path_to_txt_),sep=',', header=None)
				id_list = imagelist[1]
				imglist = imagelist[0]

				self.id_list = id_list
				self.imglist = imglist
				self.path_to_images_ = path_to_images_
				self.image_loader = default_image_loader
				self.transform = transform

		def __getitem__(self, index):
			data =  self.transform(self.image_loader(os.path.join(self.path_to_images_,self.imglist[index])))
			# label_idx = torch.from_numpy(np.array(self.id_list[index]))
			label_idx = [self.id_list[index]]
			return data, label_idx

		def __len__(self):
			return len(self.imglist)




def df_c2s_pair(temp_path):
	consumer = glob.glob('{}/comsumer*'.format(temp_path))
	num_samples_C = len(consumer)
	chosen_idx_C_ = np.random.randint(num_samples_C, size=1)[0]
	data_path1 = consumer[chosen_idx_C_]

	shop = glob.glob('{}/shop*'.format(temp_path))
	num_samples_S = len(shop)
	chosen_idx_S_ = np.random.randint(num_samples_S, size=1)[0]
	data_path2 = shop[chosen_idx_S_]

	return data_path1, data_path2

def neg_c2s(folder_consumer_shop, path_to_images, combinations_type, transform1, transform2, image_loader, index):

	temp_path = os.path.join(path_to_images,folder_consumer_shop)

	if combinations_type == 'C2S':
		data_path1, data_path2 = df_c2s_pair(temp_path)

	elif combinations_type == 'ALL':
		temp_combinations_type = ['C2S', 'Other']
		chosen_comb_idx = np.random.randint(2,size=1)[0]
		chosen_comb_type = temp_combinations_type[chosen_comb_idx]

		if chosen_comb_type == 'C2S':
			data_path1, data_path2 = df_c2s_pair(temp_path)

		elif chosen_comb_type == 'Other': 

			temp_other_combinations_type = ['C2C', 'S2S']
			chosen_other_comb_idx = np.random.randint(2,size=1)[0]
			chosen_other_comb_type = temp_other_combinations_type[chosen_other_comb_idx]

			if chosen_other_comb_type == 'C2C':
				consumer = glob.glob('{}/comsumer*'.format(temp_path))
				num_samples_C = len(consumer)
				chosen_idx_C_ = np.random.randint(num_samples_C, size=2)
				data_path1 = consumer[chosen_idx_C_[0]]
				data_path2 = consumer[chosen_idx_C_[1]]

			elif chosen_other_comb_type == 'S2S':
				shop = glob.glob('{}/shop*'.format(temp_path))
				num_samples_S = len(shop)
				chosen_idx_S_ = np.random.randint(num_samples_S, size=2)
				data_path1 = shop[chosen_idx_S_[0]]
				data_path2 = shop[chosen_idx_S_[1]]

	a_ = transform1(image_loader(data_path1))
	p_ = transform2(image_loader(data_path2))

	l1_ = torch.Tensor([index])
	l2_ = torch.Tensor([index])


	return a_, p_, l1_, l2_


def sample_set_w_both_C_S(folder_w_C_S, path_to_images_):

	idx_2_consider = []

	for jj_ in range(len(folder_w_C_S)):
		temp_path = os.path.join(path_to_images_,folder_w_C_S[jj_])

		consumer = glob.glob1(temp_path, 'comsumer*')
		shop = glob.glob1(temp_path, 'shop*')

		if len(consumer) and len(shop):
			idx_2_consider.append(jj_)

	final_folder_w_C_S = folder_w_C_S[idx_2_consider]

	return final_folder_w_C_S



class load_df_mined_dataset(Dataset):
		def __init__(self,  path_to_data='XXXX', train_test_type='XXXX', dataset_name='XXXX', path_to_images_='XXXX', 
			combinations_type='XXXX', transform1=None, transform2=None,  loader=neg_c2s, image_loader= default_image_loader):

			self.dataset_name = dataset_name

			if self.dataset_name == 'DeepFashion':


				imagelist = pd.read_csv(path_to_data, delimiter=' ', names=['image_pair_name_1', 'image_pair_name_2', 'item_id', 'evaluation_status'])

				if train_test_type =='train':
					sample_set = imagelist[imagelist['evaluation_status']=='train']
				elif train_test_type =='trainval':
					test_ids = imagelist[imagelist['evaluation_status']=='test'].index
					sample_set = imagelist.drop(test_ids)

				sample_set = sample_set.reset_index()


				#####
				#    Folders with consumer and shop images based on if the folders contain shop images.
				####

				folder_w_C_S = sample_set['image_pair_name_2']
				folder_w_C_S = pd.unique(pd.Series(folder_w_C_S).str.split(pat = "/shop_", expand=True)[0])

				# Remove path to folders with missing consumer images

				final_folder_w_C_S = sample_set_w_both_C_S(folder_w_C_S, path_to_images_)



				self.all_folders = final_folder_w_C_S
				self.image_loader = default_image_loader
				self.loader = neg_c2s
				self.path_to_images_ = path_to_images_
				self.combinations_type = combinations_type
				self.transform1 = transform1
				self.transform2 = transform2


		def __getitem__(self, index):
			anc, pos, label1, label2 = self.loader(self.all_folders[index], self.path_to_images_, self.combinations_type, self.transform1, self.transform2, self.image_loader, index)

			return anc, pos, label1, label2


		def __len__(self):
			return len(self.all_folders)



def load_crop_image(image_path,bbox):
	frame = Image.open(image_path).convert('RGB')
	X, Y, W, H = bbox['left'], bbox['top'], bbox['width'], bbox['height']
	M=0.0
	left, top, right, bottom = int(X-M*W), int(Y-M*H), int(X+(1+2*M)*W), int(Y+(1+2*M)*H)
	area = (left, top, right, bottom)
	# area = (bbox['left'], bbox['top'], bbox['width'], bbox['height'])
	image = frame.crop(area)

	return image


def df_final_c2s_pair(data):
	consumer = data['Consumer'][0][0]['path']
	data_product_path1 = data['Consumer'][0][0]['product']
	num_samples_C = len(consumer)
	chosen_idx_C_ = np.random.randint(num_samples_C, size=1)[0]
	data_path1 = consumer[chosen_idx_C_]

	shop = data['Shop'][0][0]['path']
	data_product_path2 = data['Shop'][0][0]['product']
	num_samples_S = len(shop)
	chosen_idx_S_ = np.random.randint(num_samples_S, size=1)[0]
	data_path2 = shop[chosen_idx_S_]

	# a_ = transform1(image_loader(os.path.join(path_to_df_data,data_product_path1,data_path1)))
	# p_ = transform2(image_loader(os.path.join(path_to_df_data,data_product_path2,data_path2)))

	return data_product_path1, data_product_path2, data_path1, data_path2


def final_loader_df(path_to_df_data, data, df_combinations_type, transform1, transform2, image_loader, index):


	if df_combinations_type == 'C2S':
		data_product_path1, data_product_path2, data_path1, data_path2 = df_final_c2s_pair(data)


	elif df_combinations_type == 'ALL':
		temp_combinations_type = ['C2S', 'Other']
		chosen_comb_idx = np.random.randint(2,size=1)[0]
		chosen_comb_type = temp_combinations_type[chosen_comb_idx]

		if chosen_comb_type == 'C2S':
			data_product_path1, data_product_path2, data_path1, data_path2 = df_final_c2s_pair(data)

		elif chosen_comb_type == 'Other': 

			temp_other_combinations_type = ['C2C', 'S2S']
			chosen_other_comb_idx = np.random.randint(2,size=1)[0]
			chosen_other_comb_type = temp_other_combinations_type[chosen_other_comb_idx]

			if chosen_other_comb_type == 'C2C':
				consumer = data['Consumer'][0][0]['path']
				data_product_path1 = data['Consumer'][0][0]['product']
				num_samples_C = len(consumer)
				chosen_idx_C_ = np.random.randint(num_samples_C, size=2)
				data_path1 = consumer[chosen_idx_C_[0]]
				data_path2 = consumer[chosen_idx_C_[1]]
				data_product_path2 = data_product_path1

			elif chosen_other_comb_type == 'S2S':
				shop = data['Shop'][0][0]['path']
				data_product_path1 = data['Shop'][0][0]['product']
				num_samples_S = len(shop)
				chosen_idx_S_ = np.random.randint(num_samples_S, size=2)
				data_path1 = shop[chosen_idx_S_[0]]
				data_path2 = shop[chosen_idx_S_[1]]
				data_product_path2 = data_product_path1

	a_ = transform1(image_loader(os.path.join(path_to_df_data,data_product_path1,data_path1)))
	p_ = transform2(image_loader(os.path.join(path_to_df_data,data_product_path2,data_path2)))

	l1_ = torch.Tensor([index])
	l2_ = torch.Tensor([index])
	return a_, p_, l1_, l2_



def final_loader_df_s2s(path_to_df_data, path_to_s2s_data, data, df_combinations_type, s2s_combinations_type, transform1, transform2, image_loader, index):

	# Check if the given sample is DF or S2S
	num_keys = len(data['Consumer'][0][0].keys()) # 6 --> S2S,  2 --> DF

	if num_keys == 6:
		#
		# Street2Shop
		#
		# The given sample is S2S sample

		if s2s_combinations_type == 'C2C':
			consumer = data['Consumer'][0]
			num_samples_C = len(consumer)

			chosen_idx_C_ = np.random.randint(num_samples_C, size=2)
			chosen_idx_C1_ = chosen_idx_C_[0]
			chosen_idx_C2_ = chosen_idx_C_[1]

			consumer_info1 = consumer[chosen_idx_C1_]
			consumer_path1 = os.path.join(path_to_s2s_data,consumer_info1['path'])
			bbox1 = consumer_info1['bbox']
			a_ = transform1(load_crop_image(consumer_path1,bbox1))


			consumer_info2 = consumer[chosen_idx_C2_]
			consumer_path2 = os.path.join(path_to_s2s_data,consumer_info2['path'])
			bbox2 = consumer_info2['bbox']
			p_ = transform2(load_crop_image(consumer_path1,bbox1))


		elif s2s_combinations_type == 'S2S':
			shop = data['Shop'][0]
			num_samples_S = len(shop)

			chosen_idx_S_ = np.random.randint(num_samples_S, size=2)
			chosen_idx_S1_ = chosen_idx_S_[0]
			chosen_idx_S2_ = chosen_idx_S_[1]

			shop_info1 = shop[chosen_idx_S1_]
			shop_path1 = os.path.join(path_to_s2s_data,shop_info1['path'])
			a_ = transform1(image_loader(shop_path1))

			shop_info2 = shop[chosen_idx_S2_]
			shop_path2 = os.path.join(path_to_s2s_data,shop_info2['path'])
			p_ = transform2(image_loader(shop_path2))

		l1_ = torch.Tensor([index])
		l2_ = torch.Tensor([index])

	elif num_keys == 2:
		#
		# DeepFashion
		#
		# The given sample is a DeepFashion sample
		a_, p_, l1_, l2_  = final_loader_df(path_to_df_data, data, df_combinations_type, transform1, transform2, image_loader, index)


	return a_, p_, l1_, l2_





class load_df_s2s_mined_datataset(Dataset):
		def __init__(self,  path_to_df_data='XXXX', path_to_s2s_data='XXXX', dataset_name='XXXX', path_to_images_='XXXX', 
					df_combinations_type='XXXX', s2s_combinations_type='XXXX', 
					transform1=None, transform2=None, loader=None, image_loader= default_image_loader):

			self.dataset_name = dataset_name

			data = json.load(open(path_to_images_))

			# len(s2s_json['500']['Consumer'][0][0].keys())
			# len(All_Samples['10000']['Consumer'][0][0].keys())

			# if self.dataset_name == 'DeepFashion':

			self.path_to_df_data = path_to_df_data
			self.path_to_s2s_data = path_to_s2s_data
			self.data = data

			self.image_loader = default_image_loader
			self.loader = final_loader_df_s2s

			self.df_combinations_type = df_combinations_type
			self.s2s_combinations_type = s2s_combinations_type
			self.transform1 = transform1
			self.transform2 = transform2

		def __getitem__(self, index):
			anc, pos, label1, label2 = self.loader(self.path_to_df_data, self.path_to_s2s_data, self.data['{}'.format(index)], self.df_combinations_type, 
														self.s2s_combinations_type, self.transform1, self.transform2, self.image_loader, index)

			return anc, pos, label1, label2


		def __len__(self):
			return len(self.data)



def default_s2s_test_image_loader(path_to_images_, data_, consumer_shop_, transform_):
	data_path = os.path.join(path_to_images_,data_['path'])
	product_id = data_['product']
	label = data_['label']
	category = data_['category']

	if consumer_shop_ == 'Consumer':
		bbox = data_['bbox']
		image = transform_(load_crop_image(data_path,bbox))

	elif consumer_shop_ == 'Shop':
		image = transform_(default_image_loader(data_path))

	product_id = torch.Tensor([product_id])
	label = torch.Tensor([label])

	return image, product_id, label 


def default_s2s_crop_test_image_loader(path_to_images_, data_, consumer_shop_, transform_):
	product_id = data_['product']
	label = data_['label']
	category = data_['category']
	image_path = data_['path']
	image_number = image_path.split('/')[-1].split('.')[0]
	# data_path = os.path.join(path_to_images_,data_['path'])
	data_path = '{}/{}/rtr/{}_{:07d}_rtr_{}.jpg'.format(path_to_images_,category,category,product_id,image_number)

	image = transform_(default_image_loader(data_path))

	product_id = torch.Tensor([product_id])
	label = torch.Tensor([label])

	return image, product_id, label 



class load_test_s2s_dataset(Dataset):
		def __init__(self,  path_to_images_='XXXX', path_to_txt_='XXXX', dataset_name='XXX', consumer_shop = 'XXX', full_or_crop= None, transform=None, image_loader= default_s2s_test_image_loader):

			data = json.load(open(path_to_txt_))
			self.data = data

			self.dataset_name = dataset_name
			self.consumer_shop = consumer_shop
			self.path_to_images_ = path_to_images_
			self.transform = transform
			self.full_or_crop = full_or_crop

			if self.full_or_crop == 'full':
				self.image_loader = default_s2s_test_image_loader
			elif self.full_or_crop == 'crop':
				self.image_loader = default_s2s_crop_test_image_loader


		def __getitem__(self, index):
			data, label_idx, label =  self.image_loader(self.path_to_images_,self.data[index], self.consumer_shop, self.transform)

			return data, label_idx, label

		def __len__(self):
			return len(self.data)



class load_path_feat_extraction(Dataset):
		def __init__(self,  path_to_images_='XXXX', path_to_txt_='XXXX', transform=None, image_loader= default_image_loader):

			imagelist = pd.read_csv(open(path_to_txt_),sep=' ', header=None)
			self.imglist = imagelist[0]
			self.pid = imagelist[1]
			self.cid = imagelist[2]
			self.path_to_images_ = path_to_images_
			self.image_loader = default_image_loader
			self.transform = transform

		def __getitem__(self, index):
			data =  self.transform(self.image_loader(os.path.join(self.path_to_images_,self.imglist[index])))
			pid = [self.pid[index]]
			cid = [self.cid[index]]
			return data, pid, cid

		def __len__(self):
			return len(self.imglist)

def df_c2s_c_s_pairs(data_, c_feat_, s_feat_):

	temp_anc_, temp_pos_ = [], []

	# Consumer and Shop pair
	# Consumer samples
	temp_Fc_idx =  np.asarray(data_['Fc'])
	tt_fc_1_ = temp_Fc_idx[np.random.randint(len(temp_Fc_idx), size=1)[0]]
	temp_anc_.append(torch.from_numpy(c_feat_[tt_fc_1_]).unsqueeze(0))

	# Shop samples
	temp_Fs_idx =  np.asarray(data_['Fs'])
	tt_fs_1_ = temp_Fs_idx[np.random.randint(len(temp_Fs_idx), size=1)[0]]
	temp_pos_.append(torch.from_numpy(s_feat_[tt_fs_1_]).unsqueeze(0))


	# Consumer and Consumer pair
	if len(temp_Fc_idx) > 1:
		# pdb.set_trace()
		# print('Make consumer pairs')
		tt_fc_1_2_ = temp_Fc_idx[random.sample(range(len(temp_Fc_idx)), 2)]
		temp_anc_.append(torch.from_numpy(c_feat_[tt_fc_1_2_[0]]).unsqueeze(0))
		temp_pos_.append(torch.from_numpy(c_feat_[tt_fc_1_2_[1]]).unsqueeze(0))


	# Shop and Shop pair
	if len(temp_Fs_idx) > 1:
		# pdb.set_trace()
		# print('Make shop pairs')
		tt_fs_1_2_ = temp_Fs_idx[random.sample(range(len(temp_Fs_idx)), 2)]
		temp_anc_.append(torch.from_numpy(s_feat_[tt_fs_1_2_[0]]).unsqueeze(0))
		temp_pos_.append(torch.from_numpy(s_feat_[tt_fs_1_2_[1]]).unsqueeze(0))

	return temp_anc_, temp_pos_, tt_fc_1_


def default_sample_loader(c_feat_,s_feat_,method_type, data_pairs, batch_size):

	max_num_of_samples_ = len(data_pairs)

	if max_num_of_samples_ <= batch_size:
		batch_size = max_num_of_samples_
		# temp_chosen_idx = np.random.randint(batch_size, size=1)[0]
	considered_idx_ = random.sample(range(max_num_of_samples_),batch_size)

	# anc, pos, label = [], [], []

	label = []


	for batch_idx, ii_ in enumerate(considered_idx_):

		anc_, pos_, label_ = [], [], []

		all_samples_ = data_pairs[ii_] 
		# To know what type of the data is of what type.
		temp_num_of_annotations = len(all_samples_)	

		# Unique product id, used as a labele for metric learning.
		temp_product_id = all_samples_['pid']

		if method_type == 'L1':								  							 	# L1:F 
			anc_, pos_, _ = df_c2s_c_s_pairs(all_samples_, c_feat_, s_feat_)

		if method_type=='L12' or method_type=='L123':  										# L2:FW^{~}

			if temp_num_of_annotations >= 3:
				anc_, pos_, tt_fc_1_ = df_c2s_c_s_pairs(all_samples_, c_feat_, s_feat_)
			
			if temp_num_of_annotations==4:

				# Clustering based weak shop pairs
				temp_FWs_idx =  np.asarray(all_samples_['FWs'])
				tt_fws_1_ = temp_FWs_idx[np.random.randint(len(temp_FWs_idx), size=1)[0]]

				# Fc to Ws^{~} pair
				anc_.append(torch.from_numpy(c_feat_[tt_fc_1_]).unsqueeze(0))
				pos_.append(torch.from_numpy(s_feat_[tt_fws_1_]).unsqueeze(0))

				if len(temp_FWs_idx) > 1:
					# print('Make shop pairs')
					tt_fws_1_2_ = temp_FWs_idx[random.sample(range(len(temp_FWs_idx)), 2)]
					anc_.append(torch.from_numpy(s_feat_[tt_fws_1_2_[0]]).unsqueeze(0))
					pos_.append(torch.from_numpy(s_feat_[tt_fws_1_2_[1]]).unsqueeze(0))


		if method_type=='L123' and temp_num_of_annotations==2: 								# L3:W^{~}
			# print('L123')

			temp_Ws_idx =  np.asarray(all_samples_['Ws'])
			tt_ws_1_2_ = temp_Ws_idx[random.sample(range(len(temp_Ws_idx)), 2)]
			anc_.append(torch.from_numpy(s_feat_[tt_ws_1_2_[0]]).unsqueeze(0))
			pos_.append(torch.from_numpy(s_feat_[tt_ws_1_2_[1]]).unsqueeze(0))


		label_ = (temp_product_id*np.ones(len(anc_)).astype(int)).tolist()

		if batch_idx > 0:

			anc_ = torch.cat(anc_,dim=0)
			pos_ = torch.cat(pos_,dim=0)

			anc = torch.cat((anc,anc_), dim=0)
			pos = torch.cat((pos,pos_), dim=0)

		else:
			anc = torch.cat(anc_,dim=0)
			pos = torch.cat(pos_,dim=0)

		label.append(label_)
		# pdb.set_trace()

	# anc = torch.from_numpy(c_feat_[np.asarray(list(itertools.chain.from_iterable(anc)))])
	# pos = torch.from_numpy(s_feat_[np.asarray(list(itertools.chain.from_iterable(pos)))])
	label = torch.from_numpy(np.asarray(list(itertools.chain.from_iterable(label))))
	
	return anc, pos, label

class load_MLP_data(Dataset):
		def __init__(self,  path_to_data_df_2_='XXXX', path_to_data_s2s_2_='XXXX', feat_epoch_num_=60, feat_type_2_='XXXX', num_classes=0, method_type='L1', F_FW_W_= None, batch_size=256, 
					apl_buffer=None, sample_loader=default_sample_loader ):

			########
			# Shop Feats
			########
			self.feat_epoch_num_ = feat_epoch_num_
			data_s_df_ = h5py.File(os.path.join(path_to_data_df_2_,'train_Shop_{}_{}.h5'.format(feat_type_2_, self.feat_epoch_num_)),'r')
			data_s_df_feat_ = np.asarray(data_s_df_['X_']) 

			data_s_s2s_ = h5py.File(os.path.join(path_to_data_s2s_2_,'train_Shop_{}_{}.h5'.format(feat_type_2_, self.feat_epoch_num_)),'r')
			data_s_s2s_feat_ = np.asarray(data_s_s2s_['X_']) 

			all_s_feat_ = np.concatenate((data_s_df_feat_,data_s_s2s_feat_))

			self.data_s_df_feat_ = data_s_df_feat_
			self.data_s_s2s_feat_ = data_s_s2s_feat_

			########
			# Consumer Feats
			########
			data_c_df_ = h5py.File(os.path.join(path_to_data_df_2_,'train_Consumer_{}_{}.h5'.format(feat_type_2_, self.feat_epoch_num_)),'r')
			data_c_df_feat_ = np.asarray(data_c_df_['X_']) 

			self.s_feat_ = all_s_feat_
			self.c_feat_ = data_c_df_feat_

			self.method_type = method_type
			self.num_classes = num_classes

			self.sample_loader = sample_loader
			self.F_FW_W_ = F_FW_W_
			self.batch_size = batch_size
			self.apl_buffer=apl_buffer



		def __getitem__(self, index):
			anc, pos, label =  self.sample_loader(self.c_feat_,self.s_feat_,self.method_type, self.F_FW_W_[index], self.batch_size)
			# anc, pos, label = anc.unsqueeze(0), pos.unsqueeze(0), label.unsqueeze(0)
			# print('Class: {}, Sample: {}'.format(index,anc.shape))

			self.apl_buffer[index] = []
			self.apl_buffer[index].append({'anc':anc, 'pos':pos, 'label':label})
			return self.apl_buffer, index 

		def __len__(self):
			return self.num_classes 

		def get_embedding_consumer(self):
			return self.c_feat_

		def get_embedding_shop(self):
			return self.s_feat_

		def get_embedding_df_shop(self):
			return self.data_s_df_feat_

		def get_embedding_s2s_shop(self):
			return self.data_s_s2s_feat_




class load_MLP_test_data(Dataset):
		def __init__(self,  path_to_data_='XXXX'):

			self.path_to_data_ = path_to_data_
			data_ = h5py.File(self.path_to_data_,'r')
			data_ = np.asarray(data_['X_']) 
			self.data_ = data_
			del data_

		def __getitem__(self, index):
			data =  torch.from_numpy(self.data_[index])
			return data

		def __len__(self):
			return len(self.data_) 



class load_MLP_CURR_train_data(Dataset):
		def __init__(self,  feats=None):

			self.data_ = feats

		def __getitem__(self, index):
			data =  torch.from_numpy(self.data_[index])
			return data

		def __len__(self):
			return len(self.data_) 

###################
#
#
# FULL ID - Training full model within class
#
#
###################

def FULL_df_c2s_c_s_pairs(data_, c_feat_, s_feat_):

	temp_anc_, temp_pos_ = [], []

	# Consumer and Shop pair
	# Consumer samples
	temp_Fc_idx =  np.asarray(data_['Fc'])
	tt_fc_1_ = temp_Fc_idx[np.random.randint(len(temp_Fc_idx), size=1)[0]]
	temp_anc_.append(c_feat_[tt_fc_1_])

	# Shop samples
	temp_Fs_idx =  np.asarray(data_['Fs'])
	tt_fs_1_ = temp_Fs_idx[np.random.randint(len(temp_Fs_idx), size=1)[0]]
	temp_pos_.append(s_feat_[tt_fs_1_])


	# Consumer and Consumer pair
	if len(temp_Fc_idx) > 1:
		# pdb.set_trace()
		# print('Make consumer pairs')
		tt_fc_1_2_ = temp_Fc_idx[random.sample(range(len(temp_Fc_idx)), 2)]
		temp_anc_.append(c_feat_[tt_fc_1_2_[0]])
		temp_pos_.append(c_feat_[tt_fc_1_2_[1]])


	# Shop and Shop pair
	if len(temp_Fs_idx) > 1:
		# pdb.set_trace()
		# print('Make shop pairs')
		tt_fs_1_2_ = temp_Fs_idx[random.sample(range(len(temp_Fs_idx)), 2)]
		temp_anc_.append(s_feat_[tt_fs_1_2_[0]])
		temp_pos_.append(s_feat_[tt_fs_1_2_[1]])

	return temp_anc_, temp_pos_, tt_fc_1_


def FULL_default_sample_loader(c_feat_,s_feat_,method_type, data_pairs, batch_size, transform1, transform2, anchor, positive):


	max_num_of_samples_ = len(data_pairs)

	if max_num_of_samples_ <= batch_size:
		batch_size = max_num_of_samples_
		# temp_chosen_idx = np.random.randint(batch_size, size=1)[0]
	considered_idx_ = random.sample(range(max_num_of_samples_),batch_size)

	anc, pos, label = [], [], []

	for batch_idx, ii_ in enumerate(considered_idx_):
		# ii_ = 100

		anc_, pos_, label_ = [], [], []

		all_samples_ = data_pairs[ii_] 

		# To know what type of the data is of what type.
		temp_num_of_annotations = len(all_samples_)	

		# Unique product id, used as a labele for metric learning.
		temp_product_id = all_samples_['pid']

		if method_type == 'L1':								  							 	# L1:F 
			anc_, pos_, _ = FULL_df_c2s_c_s_pairs(all_samples_, c_feat_, s_feat_)

		if method_type=='L12':  										# L2:FW^{~}

			if temp_num_of_annotations >= 3:
				anc_, pos_, tt_fc_1_ = FULL_df_c2s_c_s_pairs(all_samples_, c_feat_, s_feat_)

			
			if temp_num_of_annotations==4:

				# Clustering based weak shop pairs
				temp_FWs_idx =  np.asarray(all_samples_['FWs'])
				tt_fws_1_ = temp_FWs_idx[np.random.randint(len(temp_FWs_idx), size=1)[0]]

				# Fc to Ws^{~} pair
				anc_.append(c_feat_[tt_fc_1_])
				pos_.append(s_feat_[tt_fws_1_])

				if len(temp_FWs_idx) > 1:
					# print('Make shop pairs')
					tt_fws_1_2_ = temp_FWs_idx[random.sample(range(len(temp_FWs_idx)), 2)]
					anc_.append(s_feat_[tt_fws_1_2_[0]])
					pos_.append(s_feat_[tt_fws_1_2_[1]])


		label_ = (temp_product_id*np.ones(len(anc_)).astype(int)).tolist()

		anc.append(anc_)
		pos.append(pos_)
		label.append(label_)

	anc = np.asarray(list(itertools.chain.from_iterable(anc)))
	pos = np.asarray(list(itertools.chain.from_iterable(pos)))
	label = torch.from_numpy(np.asarray(list(itertools.chain.from_iterable(label))))


	# Picking up an exact batch size with 32 samples
	considered_batch_idx_ = random.sample(range(len(anc)),batch_size)

	anc = anc[considered_batch_idx_]
	pos = pos[considered_batch_idx_]
	label = label[considered_batch_idx_]


	for ii_ in range(len(anc)):
		anchor[ii_,:,:,:] = transform1(default_image_loader(anc[ii_]))
		positive[ii_,:,:,:] = transform2(default_image_loader(pos[ii_]))

	return anchor, positive, label


class load_FULL_data(Dataset):
		def __init__(self, path_to_images_df_='XXXX', path_to_images_s2s_='XXXX', path_to_consumer_df_txt_='XXXX', path_to_shop_df_txt_='XXXX',
							path_to_shop_s2s_txt_='XXXX',  num_classes=0, method_type='L12', F_FW_= None, batch_size=32,
							apl_buffer=None, transform1=None, transform2=None, sample_loader=FULL_default_sample_loader ):

			########
			# Shop 
			########
			data_s_df_ = path_to_images_df_ + pd.read_csv(open(path_to_shop_df_txt_),sep=' ', header=None)[0]
			data_s_s2s_ = path_to_images_s2s_ + pd.read_csv(open(path_to_shop_s2s_txt_),sep=' ', header=None)[0]
			all_data_s_ = np.asarray(pd.concat([data_s_df_,data_s_s2s_]))

			########
			# Consumer 
			########
			data_c_df_ = path_to_images_df_ + pd.read_csv(open(path_to_consumer_df_txt_),sep=' ', header=None)[0]
	
			self.all_s_paths_ = all_data_s_
			self.df_c_paths_  = data_c_df_

			self.method_type = method_type
			self.num_classes = num_classes

			self.sample_loader = sample_loader
			self.F_FW_ = F_FW_
			self.batch_size = batch_size
			self.apl_buffer=apl_buffer
			self.transform1 = transform1
			self.transform2 = transform2
			self.anchor = torch.zeros((batch_size,3,224,224))
			self.positive = torch.zeros((batch_size,3,224,224))



		def __getitem__(self, index):

			anc, pos, label =  self.sample_loader(self.df_c_paths_,self.all_s_paths_,self.method_type, self.F_FW_[index], self.batch_size, self.transform1, self.transform2, self.anchor, self.positive)
			# anc, pos, label = anc.unsqueeze(0), pos.unsqueeze(0), label.unsqueeze(0)
			# print('Class: {}, Sample: {}'.format(index,anc.shape))

			self.apl_buffer[index] = []
			self.apl_buffer[index].append({'anc':anc, 'pos':pos, 'label':label})
			return self.apl_buffer, index 

		def __len__(self):
			return self.num_classes



###################
#
#
# Weighting based datatet prepartion
#
#
###################
def default_Weighting_sample_loader(c_feat_,s_feat_,method_type, data_pairs, batch_size):

	max_num_of_samples_ = len(data_pairs)

	if max_num_of_samples_ <= batch_size:
		batch_size = max_num_of_samples_
		# temp_chosen_idx = np.random.randint(batch_size, size=1)[0]
	considered_idx_ = random.sample(range(max_num_of_samples_),batch_size)

	# anc, pos, label = [], [], []

	label = []


	for batch_idx, ii_ in enumerate(considered_idx_):

		anc_, pos_, label_ = [], [], []

		all_samples_ = data_pairs[ii_] 
		# To know what type of the data is of what type.
		temp_num_of_annotations = len(all_samples_)	

		# Unique product id, used as a labele for metric learning.
		temp_product_id = all_samples_['pid']

		if method_type == 'L1':								  							 	# L1:F 
			anc_, pos_, _ = df_c2s_c_s_pairs(all_samples_, c_feat_, s_feat_)

		if method_type=='L12':  															# L2:FW^{~}

			if temp_num_of_annotations >= 3:
				anc_, pos_, tt_fc_1_ = df_c2s_c_s_pairs(all_samples_, c_feat_, s_feat_)
			
			if temp_num_of_annotations==4:

				# Clustering based weak shop pairs
				temp_FWs_idx =  np.asarray(all_samples_['FWs'])

				# ## #If more than one sample, choose the closest
				# if len(temp_FWs_idx) > 1:
				# 	# Obtain all S2S samples, l2norm, compute score between given DF-sample and S2S samples, and use the pair with minimum score.
				# 	##### V: Consumer-based
				# 	# cons_ = F.normalize(torch.from_numpy(c_feat_[tt_fc_1_]).unsqueeze(0), p=2, dim=1)#.numpy()
				# 	# sho_ = F.normalize(torch.from_numpy(s_feat_[temp_FWs_idx]), p=2, dim=1)#.numpy()  # L2 normalize
				# 	# mm = torch.mm(sho_,cons_.transpose(0,1))
				# 	# _, pred_ = mm.max(0)

				# 	##### S: Shop-based
				#	temp_Fs_idx =  np.asarray(all_samples_['Fs'])
				#	tt_fs_1_ = temp_Fs_idx[np.random.randint(len(temp_Fs_idx), size=1)[0]]

				#	df_sho_ = F.normalize(torch.from_numpy(s_feat_[tt_fs_1_]).unsqueeze(0), p=2, dim=1)
				#	sho_ = F.normalize(torch.from_numpy(s_feat_[temp_FWs_idx]), p=2, dim=1)
				#	mm = torch.mm(sho_,df_sho_.transpose(0,1))
				#	_, pred_ = mm.max(0)

				#	# Fc to Ws^{~} pair
				#	anc_.append(torch.from_numpy(c_feat_[tt_fc_1_]).unsqueeze(0))
				#	pos_.append(torch.from_numpy(s_feat_[temp_FWs_idx[pred_]]).unsqueeze(0))
				# 	pdb.set_trace()
				# else:
				
				# ####### G's  -- Comment if part of V and S
				tt_fws_1_ = temp_FWs_idx[np.random.randint(len(temp_FWs_idx), size=1)[0]]

				# Fc to Ws^{~} pair
				anc_.append(torch.from_numpy(c_feat_[tt_fc_1_]).unsqueeze(0))
				pos_.append(torch.from_numpy(s_feat_[tt_fws_1_]).unsqueeze(0))

				# Make weak Shop2Shop pairs
				if len(temp_FWs_idx) > 1:
					# print('Make shop pairs')
					tt_fws_1_2_ = temp_FWs_idx[random.sample(range(len(temp_FWs_idx)), 2)]
					anc_.append(torch.from_numpy(s_feat_[tt_fws_1_2_[0]]).unsqueeze(0))
					pos_.append(torch.from_numpy(s_feat_[tt_fws_1_2_[1]]).unsqueeze(0))



		label_ = (temp_product_id*np.ones(len(anc_)).astype(int)).tolist()

		if batch_idx > 0:

			anc_ = torch.cat(anc_,dim=0)
			pos_ = torch.cat(pos_,dim=0)

			anc = torch.cat((anc,anc_), dim=0)
			pos = torch.cat((pos,pos_), dim=0)

		else:
			anc = torch.cat(anc_,dim=0)
			pos = torch.cat(pos_,dim=0)

		label.append(label_)
		# pdb.set_trace()

	# anc = torch.from_numpy(c_feat_[np.asarray(list(itertools.chain.from_iterable(anc)))])
	# pos = torch.from_numpy(s_feat_[np.asarray(list(itertools.chain.from_iterable(pos)))])
	label = torch.from_numpy(np.asarray(list(itertools.chain.from_iterable(label))))
	
	return anc, pos, label

class load_Weighting_MLP_data(Dataset):
		def __init__(self,  path_to_data_df_2_='XXXX', path_to_data_s2s_2_='XXXX', feat_epoch_num_=60, feat_type_2_='XXXX', num_classes=0, method_type='L1', F_FW_W_= None, batch_size=256, 
					apl_buffer=None, sample_loader=default_Weighting_sample_loader ):

			########
			# Shop Feats
			########
			self.feat_epoch_num_ = feat_epoch_num_
			data_s_df_ = h5py.File(os.path.join(path_to_data_df_2_,'train_Shop_{}_{}.h5'.format(feat_type_2_, self.feat_epoch_num_)),'r')
			data_s_df_feat_ = np.asarray(data_s_df_['X_']) 

			data_s_s2s_ = h5py.File(os.path.join(path_to_data_s2s_2_,'train_Shop_{}_{}.h5'.format(feat_type_2_, self.feat_epoch_num_)),'r')
			data_s_s2s_feat_ = np.asarray(data_s_s2s_['X_']) 

			all_s_feat_ = np.concatenate((data_s_df_feat_,data_s_s2s_feat_))

			self.data_s_df_feat_ = data_s_df_feat_
			self.data_s_s2s_feat_ = data_s_s2s_feat_

			########
			# Consumer Feats
			########
			data_c_df_ = h5py.File(os.path.join(path_to_data_df_2_,'train_Consumer_{}_{}.h5'.format(feat_type_2_, self.feat_epoch_num_)),'r')
			data_c_df_feat_ = np.asarray(data_c_df_['X_']) 

			self.s_feat_ = all_s_feat_
			self.c_feat_ = data_c_df_feat_

			self.method_type = method_type
			self.num_classes = num_classes

			self.sample_loader = sample_loader
			self.F_FW_W_ = F_FW_W_
			self.batch_size = batch_size
			self.apl_buffer=apl_buffer



		def __getitem__(self, index):
			anc, pos, label =  self.sample_loader(self.c_feat_,self.s_feat_,self.method_type, self.F_FW_W_[index], self.batch_size)
			# anc, pos, label = anc.unsqueeze(0), pos.unsqueeze(0), label.unsqueeze(0)
			# print('Class: {}, Sample: {}'.format(index,anc.shape))

			self.apl_buffer[index] = []
			self.apl_buffer[index].append({'anc':anc, 'pos':pos, 'label':label})
			return self.apl_buffer, index 

		def __len__(self):
			return self.num_classes 

		def get_embedding_consumer(self):
			return self.c_feat_

		def get_embedding_shop(self):
			return self.s_feat_

		def get_embedding_df_shop(self):
			return self.data_s_df_feat_

		def get_embedding_s2s_shop(self):
			return self.data_s_s2s_feat_

######################################



def final_loader_s2s( path_to_s2s_data, data,  s2s_combinations_type, transform1, transform2, image_loader, index):

	# Check if the given sample is DF or S2S
	num_keys = len(data['Consumer'][0][0].keys()) # 6 --> S2S,  2 --> DF

	if num_keys == 6:
		#
		# Street2Shop
		#
		# The given sample is S2S sample

		if s2s_combinations_type == 'ALL':
			temp_combinations_type = ['C2S', 'Other']
			chosen_comb_idx = np.random.randint(2,size=1)[0]
			chosen_comb_type = temp_combinations_type[chosen_comb_idx]

			if chosen_comb_type == 'C2S':
				# Consumer
				consumer = data['Consumer'][0]
				num_samples_C = len(consumer)
				chosen_idx_C_ = np.random.randint(num_samples_C, size=1)
				chosen_idx_C1_ = chosen_idx_C_[0]

				consumer_info1 = consumer[chosen_idx_C1_]
				consumer_path1 = os.path.join(path_to_s2s_data,consumer_info1['path'])
				bbox1 = consumer_info1['bbox']
				a_ = transform1(load_crop_image(consumer_path1,bbox1))

				# Shop
				shop = data['Shop'][0]
				num_samples_S = len(shop)
				chosen_idx_S_ = np.random.randint(num_samples_S, size=1)
				chosen_idx_S1_ = chosen_idx_S_[0]

				shop_info1 = shop[chosen_idx_S1_]
				shop_path1 = os.path.join(path_to_s2s_data,shop_info1['path'])
				p_ = transform2(image_loader(shop_path1))


			elif chosen_comb_type == 'Other': 
				temp_other_combinations_type = ['C2C', 'S2S']
				chosen_other_comb_idx = np.random.randint(2,size=1)[0]
				chosen_other_comb_type = temp_other_combinations_type[chosen_other_comb_idx]


				if chosen_other_comb_type == 'C2C':
					# C2C
					consumer = data['Consumer'][0]
					num_samples_C = len(consumer)

					chosen_idx_C_ = np.random.randint(num_samples_C, size=2)
					chosen_idx_C1_ = chosen_idx_C_[0]
					chosen_idx_C2_ = chosen_idx_C_[1]

					consumer_info1 = consumer[chosen_idx_C1_]
					consumer_path1 = os.path.join(path_to_s2s_data,consumer_info1['path'])
					bbox1 = consumer_info1['bbox']
					a_ = transform1(load_crop_image(consumer_path1,bbox1))


					consumer_info2 = consumer[chosen_idx_C2_]
					consumer_path2 = os.path.join(path_to_s2s_data,consumer_info2['path'])
					bbox2 = consumer_info2['bbox']
					p_ = transform2(load_crop_image(consumer_path1,bbox1))


				elif chosen_other_comb_type == 'S2S':
					shop = data['Shop'][0]
					num_samples_S = len(shop)

					chosen_idx_S_ = np.random.randint(num_samples_S, size=2)
					chosen_idx_S1_ = chosen_idx_S_[0]
					chosen_idx_S2_ = chosen_idx_S_[1]

					shop_info1 = shop[chosen_idx_S1_]
					shop_path1 = os.path.join(path_to_s2s_data,shop_info1['path'])
					a_ = transform1(image_loader(shop_path1))

					shop_info2 = shop[chosen_idx_S2_]
					shop_path2 = os.path.join(path_to_s2s_data,shop_info2['path'])
					p_ = transform2(image_loader(shop_path2))

			l1_ = torch.Tensor([index])
			l2_ = torch.Tensor([index])



	return a_, p_, l1_, l2_





class load_s2s_mined_datataset(Dataset):
		def __init__(self,  path_to_s2s_data='XXXX', dataset_name='XXXX', path_to_images_='XXXX', 
					s2s_combinations_type='XXXX', 
					transform1=None, transform2=None, loader=None, image_loader= default_image_loader):

			self.dataset_name = dataset_name

			data = json.load(open(path_to_images_))

			# len(s2s_json['500']['Consumer'][0][0].keys())
			# len(All_Samples['10000']['Consumer'][0][0].keys())

			# if self.dataset_name == 'DeepFashion':

			self.path_to_s2s_data = path_to_s2s_data
			self.data = data

			self.image_loader = default_image_loader
			self.loader = final_loader_s2s

			self.s2s_combinations_type = s2s_combinations_type
			self.transform1 = transform1
			self.transform2 = transform2

		def __getitem__(self, index):
			anc, pos, label1, label2 = self.loader(self.path_to_s2s_data, self.data['{}'.format(index)],  
														self.s2s_combinations_type, self.transform1, self.transform2, self.image_loader, index)

			return anc, pos, label1, label2


		def __len__(self):
			return len(self.data)