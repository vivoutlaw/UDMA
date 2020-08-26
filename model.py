import torch
import torch.nn as nn
import torch.nn.functional as F



def l2_normalize(x, axis=-1):
	x = F.normalize(x, p=2, dim=axis)
	return x

############################
#
# MLP
#
############################

class KN_MLP(nn.Module):
	def __init__(self, naver_model='XXX', in_dim=2048, h_dim1=512, h_dim2=512):
		super(KN_MLP, self).__init__()

		# for param in naver_model.parameters():
		# 	param.requires_grad = False

		naver_model.module.fc =  nn.Sequential(
			nn.Linear(in_dim, h_dim1),
			nn.BatchNorm1d(h_dim1),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(h_dim1, h_dim2)
			# nn.BatchNorm1d(h_dim2),
			# nn.ReLU(),
			# nn.Dropout(p=0.5)
			)

		self.net = naver_model

		# self.classifier = nn.Linear(h_dim2, out_dim)

	def forward(self, x1, x2, x3):
		"""
		Note: image dimension conversion will be handled by external methods
		"""
		_ , o1 = self.net(x1)
		_ , o2 = self.net(x2)
		_ , o3 = self.net(x3)

		return o1, o2, o3

	def get_embedding(self, x):
		gmp_x, x_o = self.net(x)
		return gmp_x, x_o



class NAVER_NET(nn.Module):
	def __init__(self, naver_model='XXX'):
		super(NAVER_NET, self).__init__()

		# for param in naver_model.parameters():
		# 	param.requires_grad = False

		self.net = naver_model

	def forward(self, x1, x2, x3):
		"""
		Note: image dimension conversion will be handled by external methods
		"""
		o1 = self.net(x1)
		o2 = self.net(x2)
		o3 = self.net(x3)

		return o1, o2, o3

	def get_embedding(self, x):
		x_o = self.net(x)
		return x_o





class NAVER_MLP(nn.Module):
	def __init__(self, naver_model='XXX'):
		super(NAVER_MLP, self).__init__()

		# for param in naver_model.parameters():
		# 	param.requires_grad = False
		fc = nn.Linear(2048,2048)

		fc.weight.data.copy_(naver_model.module.fc.weight.data)
		fc.bias.data.copy_(naver_model.module.fc.bias.data)
		self.fc = fc

	def forward(self, x):
		"""
		Note: image dimension conversion will be handled by external methods
		"""
		x = self.fc(x)
		x = l2_normalize(x, axis=-1)

		return x

	def get_embedding(self, x):
		x = self.fc(x)
		return x



class NAVER_MLP_Test(nn.Module):
	def __init__(self):
		super(NAVER_MLP_Test, self).__init__()

		# for param in naver_model.parameters():
		# 	param.requires_grad = False
		fc = nn.Linear(2048,2048)

		self.fc = fc

	def forward(self, x):
		"""
		Note: image dimension conversion will be handled by external methods
		"""
		x = self.fc(x)
		x = l2_normalize(x, axis=-1)
		return x

	def get_embedding(self, x):
		x = self.fc(x)
		return x


