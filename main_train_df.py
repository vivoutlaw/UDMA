from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision import transforms
from utils import progress_bar, adjust_learning_rate
from datasets import load_df_mined_dataset
import pdb, os
import pandas as pd
import torch.nn as nn
from dirtorch.utils import common
import dirtorch.nets as nets
from pytorch_metric_learning import miners, losses
import numpy as np



############################
#
# Training and Testing
#
############################

# Training
def train_mined_net(epoch, train_test_loader):
	print('\nEpoch: %d' % epoch)
	net.train()

	for batch_idx, (anc, pos, label1, label2) in enumerate(train_test_loader):


		#for name, params in net.named_parameters():
		#	print ('{} -- {}'.format(name,params.requires_grad))

		data = torch.cat((anc, pos), dim=0).to(device)
		labels = torch.cat((label1, label2), dim=0).squeeze(1)

		# Data and labels
		data = data.to(device)

		optimizer.zero_grad()
		embeddings = net(data)
		#pdb.set_trace()
		hard_pairs = miner(embeddings, labels)
		loss = loss_func(embeddings, labels, hard_pairs)

		loss.backward()
		optimizer.step()

		# Issues with pytorch. Necessary to clear the cache.
		del data, embeddings
		torch.cuda.empty_cache()

		if batch_idx%50==0 or batch_idx+1%total_train_step==0:
			print ('Training. Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'.format(epoch+1, num_epochs, batch_idx+1, total_train_step, loss.item()))

	if np.mod(epoch+1,adjust_learning_rate_step) ==0 or np.mod(epoch+1,num_epochs)==0:
		output_dir = os.path.join(checkpoint_path,dataset_name)
		# # Save checkpoint.
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

		torch.save(state,'{}_{}_ckpt.t7'.format(checkpoint_file_name,epoch))



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


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Full Model Training')
	parser.add_argument('--batch-size', type=int, default=32, metavar='N',
											help='input batch size for training (default: 128)')
	parser.add_argument('--num-threads', type=int, default=4, metavar='N',
											help='number of threads')
	parser.add_argument('--epochs', type=int, default=45, metavar='N',
											help='number of epochs to train (default: 10)')
	parser.add_argument('--load-epoch-num', type=int, default=60, metavar='N',
											help='Load pre-trained epoch number')
	parser.add_argument('--model', type=str, default='mlp',
											help='mlp')
	parser.add_argument('--optimizer', type=str, default='ADAM',
											help='ADM | SGD')
	parser.add_argument('--df-comb', type=str, default='ALL',
											help='ALL | C2S')
	parser.add_argument('--s2s-comb', type=str, default='XXX',
											help='C2C | S2S')
	parser.add_argument('--lr', type=float, default=0.01,
											help='learning rate')
	parser.add_argument('--dataset', type=str, default='DeepFashion',
											help='DeepFashion')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
											help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
											help='how many batches to wait before logging training status')
	parser.add_argument('--resume', '-r', action='store_true')
	parser.add_argument('--checkpoint', type=str, default='../dirtorch/data/Resnet101-TL-GeM.pt',
											 help='path to weights') # ../dirtorch/data/Resnet50-AP-GeM.pt

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


	if args.optimizer == 'ADAM':
		learning_rate_ = args.lr #1e-4
		num_epochs = args.epochs #21
	elif args.optimizer == 'SGD':
		learning_rate_ = args.lr
		num_epochs = args.epochs
	
	adjust_learning_rate_step  = int(num_epochs/3)

	combinations_type = args.df_comb #'ALL' # Consumer2Shop(C2S), Shop2Shop(S2S), Consumer2Consumer(C2C), ALL: (C2S, C2C, S2S)
	batch_size = args.batch_size

	dataset_name = args.dataset
	checkpoint_path = 'models'

	#############################
	#
	# Paths
	#
	#############################

	path_to_images_df_ = '/cvhci/data/fashion_NAVER/DeepFashion/Consumer-to-shop_Clothes_Retrieval_Benchmark/Consumer-to-shop_Clothes_Retrieval_Benchmark/cropped_img'

	if dataset_name == 'DeepFashion':
		path_to_data = 'dataset_files/df_train/V_list_eval_partition.txt'


	#############################
	#
	# Dataloader
	#
	#############################
	# net.preprocess
	#  {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'input_size': 224}
	output_size_ = 224

	transform_train_1 = transforms.Compose([
		transforms.RandomRotation(45),
		transforms.Resize(256),
		transforms.RandomCrop(output_size_),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	transform_train_2 = transforms.Compose([
		transforms.RandomRotation(45),
		transforms.Resize(256),
		transforms.RandomCrop(output_size_),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	transform_train_3 = transforms.Compose([
		transforms.RandomRotation(45),
		transforms.Resize(256),
		transforms.RandomCrop(output_size_),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	if dataset_name =='DeepFashion':
		train_test_type = 'trainval' # train | trainval
		##############
		#
		# Batch mined pairs : DF dataset
		#
		##############
		trainset = load_df_mined_dataset(path_to_data=path_to_data, train_test_type=train_test_type, dataset_name=dataset_name,   
								path_to_images_=path_to_images_df_, 
								combinations_type=combinations_type, transform1=transform_train_1, transform2=transform_train_2)

	# Data loader
	trainloader = torch.utils.data.DataLoader(dataset=trainset,#batch_size=1)
											 batch_size=batch_size,
											 shuffle=True,
											 num_workers=args.num_threads,
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
	miner = miners.BatchHardMiner().to(device)
	loss_func = losses.TripletMarginLoss(margin=0.3).to(device)

	if args.optimizer == 'ADAM':
		optimizer = optim.Adam(net.parameters(), lr=learning_rate_)#1e-3)
	elif args.optimizer == 'SGD':
		optimizer = optim.SGD(net.parameters(), lr=learning_rate_, momentum=0.9, weight_decay=5e-4, nesterov=True)


	#############################
	#
	# Resume 
	#
	#############################
	if dataset_name =='DeepFashion':
		checkpoint_file_name = '{}/{}/{}_{}_{}'.format(checkpoint_path,dataset_name,dataset_name,args.optimizer,combinations_type)

	if args.resume:
		# Load checkpoint.
		load_epoch_num = args.load_epoch_num - 1
		checkpoint_number = '{}_{}_ckpt.t7'.format(checkpoint_file_name,load_epoch_num)
		print(checkpoint_number)
		assert os.path.isdir('models'), 'Error: no checkpoint directory found!'
		checkpoint = torch.load(checkpoint_number)
		net.load_state_dict(checkpoint['state_dict'])
		net =  net.to(device)
		epoch = checkpoint['epoch']
		if args.optimizer == 'ADAM':
			optimizer.load_state_dict(checkpoint['optimizer'])   
	else:
		net =  net.to(device)


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
		adjust_learning_rate(optimizer, epoch, init_lr=learning_rate_, step=adjust_learning_rate_step, decay=0.1)
		
		# train_net(epoch, train_test_loader)
		train_mined_net(epoch, train_test_loader)


