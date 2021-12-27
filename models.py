import torch
import torchvision.models as models
import torchvision
import torch.nn.functional as F
from torch import nn, Tensor

import numpy as np
from scipy import stats
from tqdm import tqdm
import os
import math
import csv
import copy
import json
from typing import Optional, List

import data_loader
from transformers import Transformer
from posencode import PositionEmbeddingSine


class L2pooling(nn.Module):
	def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
		super(L2pooling, self).__init__()
		self.padding = (filter_size - 2 )//2
		self.stride = stride
		self.channels = channels
		a = np.hanning(filter_size)[1:-1]
		g = torch.Tensor(a[:,None]*a[None,:])
		g = g/torch.sum(g)
		self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))
	def forward(self, input):
		input = input**2
		out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
		return (out+1e-12).sqrt()
	
class Net(nn.Module):
	def __init__(self,cfg,device):
		super(Net, self).__init__()
		
		
		self.device = device
		
		self.cfg = cfg
		self.L2pooling_l1 = L2pooling(channels=256)
		self.L2pooling_l2 = L2pooling(channels=512)
		self.L2pooling_l3 = L2pooling(channels=1024)
		self.L2pooling_l4 = L2pooling(channels=2048)
		


			
		if cfg.network =='resnet50':
			from resnet_modify  import resnet50 as resnet_modifyresnet
			dim_modelt = 3840
			modelpretrain = models.resnet50(pretrained=True)

		elif cfg.network =='resnet34':
			from resnet_modify  import resnet34 as resnet_modifyresnet
			modelpretrain = models.resnet34(pretrained=True)
			dim_modelt = 960
			self.L2pooling_l1 = L2pooling(channels=64)
			self.L2pooling_l2 = L2pooling(channels=128)
			self.L2pooling_l3 = L2pooling(channels=256)
			self.L2pooling_l4 = L2pooling(channels=512)
		elif cfg.network == 'resnet18':
			from resnet_modify  import resnet18 as resnet_modifyresnet
			modelpretrain = models.resnet18(pretrained=True)
			dim_modelt = 960
			self.L2pooling_l1 = L2pooling(channels=64)
			self.L2pooling_l2 = L2pooling(channels=128)
			self.L2pooling_l3 = L2pooling(channels=256)
			self.L2pooling_l4 = L2pooling(channels=512)


		torch.save(modelpretrain.state_dict(), 'modelpretrain')
		
		self.model = resnet_modifyresnet()
		self.model.load_state_dict(torch.load('modelpretrain'), strict=True)

		self.dim_modelt = dim_modelt

		os.remove("modelpretrain")
		



		nheadt=cfg.nheadt
		num_encoder_layerst=cfg.num_encoder_layerst
		dim_feedforwardt=cfg.dim_feedforwardt
		ddropout=0.5
		normalize =True
			
			
		self.transformer = Transformer(d_model=dim_modelt,nhead=nheadt,
									   num_encoder_layers=num_encoder_layerst,
									   dim_feedforward=dim_feedforwardt,
									   normalize_before=normalize,
									   dropout = ddropout)
		

		self.position_embedding = PositionEmbeddingSine(dim_modelt // 2, normalize=True)




		self.fc2 = nn.Linear(dim_modelt, self.model.fc.in_features) 
		self.fc = nn.Linear(self.model.fc.in_features*2, 1) 

		
		
		self.ReLU = nn.ReLU()
		self.avg7 = nn.AvgPool2d((7, 7))
		self.avg8 = nn.AvgPool2d((8, 8))
		self.avg4 = nn.AvgPool2d((4, 4))
		self.avg2 = nn.AvgPool2d((2, 2))
		
			   
		
		self.drop2d = nn.Dropout(p=0.1)
		self.consistency = nn.L1Loss()
		
		
		
		

	def forward(self, x):
		self.pos_enc_1 = self.position_embedding(torch.ones(1, self.dim_modelt, 7, 7).to(self.device))
		self.pos_enc = self.pos_enc_1.repeat(x.shape[0],1,1,1).contiguous()

		out,layer1,layer2,layer3,layer4 = self.model(x) 

		layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1,dim=1, p=2))))
		layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2,dim=1, p=2))))
		layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3,dim=1, p=2))))
		layer4_t =           self.drop2d(self.L2pooling_l4(F.normalize(layer4,dim=1, p=2)))
		layers = torch.cat((layer1_t,layer2_t,layer3_t,layer4_t),dim=1)

		out_t_c = self.transformer(layers,self.pos_enc)
		out_t_o = torch.flatten(self.avg7(out_t_c),start_dim=1)
		out_t_o = self.fc2(out_t_o)
		layer4_o = self.avg7(layer4)
		layer4_o = torch.flatten(layer4_o,start_dim=1)
		predictionQA = self.fc(torch.flatten(torch.cat((out_t_o,layer4_o),dim=1),start_dim=1))
		
		# =============================================================================
		# =============================================================================


		fout,flayer1,flayer2,flayer3,flayer4 = self.model(torch.flip(x, [3])) 
		flayer1_t = self.avg8( self.L2pooling_l1(F.normalize(flayer1,dim=1, p=2)))
		flayer2_t = self.avg4( self.L2pooling_l2(F.normalize(flayer2,dim=1, p=2)))
		flayer3_t = self.avg2( self.L2pooling_l3(F.normalize(flayer3,dim=1, p=2)))
		flayer4_t =            self.L2pooling_l4(F.normalize(flayer4,dim=1, p=2))
		flayers = torch.cat((flayer1_t,flayer2_t,flayer3_t,flayer4_t),dim=1)
		fout_t_c = self.transformer(flayers,self.pos_enc)
		fout_t_o = torch.flatten(self.avg7(fout_t_c),start_dim=1)
		fout_t_o = (self.fc2(fout_t_o))
		flayer4_o = self.avg7(flayer4)
		flayer4_o = torch.flatten(flayer4_o,start_dim=1)
		fpredictionQA =  (self.fc(torch.flatten(torch.cat((fout_t_o,flayer4_o),dim=1),start_dim=1)))

		
		consistloss1 = self.consistency(out_t_c,fout_t_c.detach())
		consistloss2 = self.consistency(layer4,flayer4.detach())
		consistloss = 1*(consistloss1+consistloss2)
				
		return predictionQA, consistloss


class TReS(object):
	
	def __init__(self, config, device,  svPath, datapath, train_idx, test_idx,Net):
		super(TReS, self).__init__()
		
		self.device = device
		self.epochs = config.epochs
		self.test_patch_num = config.test_patch_num
		self.l1_loss = torch.nn.L1Loss()
		self.lr = 2e-5
		self.lrratio = 10
		self.weight_decay = config.weight_decay
		self.net = Net(config,device).to(device)    
		self.droplr = config.droplr
		self.config = config
		self.clsloss =  nn.CrossEntropyLoss()
		self.paras = [{'params': self.net.parameters(), 'lr': self.lr} ]
		self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)


		train_loader = data_loader.DataLoader(config.dataset, datapath, 
											  train_idx, config.patch_size, 
											  config.train_patch_num, 
											  batch_size=config.batch_size, istrain=True)
		
		test_loader = data_loader.DataLoader(config.dataset, datapath,
											 test_idx, config.patch_size,
											 config.test_patch_num, istrain=False)
		
		self.train_data = train_loader.get_data()
		self.test_data = test_loader.get_data()


		
		
	def train(self,seed,svPath):
		best_srcc = 0.0
		best_plcc = 0.0
		print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tLearning_Rate\tdroplr')
		steps = 0
		results = {}
		performPath = svPath +'/' + 'PLCC_SRCC_'+str(self.config.vesion)+'_'+str(seed)+'.json'
		with open(performPath, 'w') as json_file2:
			json.dump(  {} , json_file2)
		
		for epochnum in range(self.epochs):
			self.net.train()
			epoch_loss = []
			pred_scores = []
			gt_scores = []
			pbar = tqdm(self.train_data, leave=False)

			for img, label in pbar:
				img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
				label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

				steps+=1
				
				self.net.zero_grad()

				pred,closs = self.net(img)
				pred2,closs2 = self.net(torch.flip(img, [3]))  

				pred_scores = pred_scores + pred.flatten().cpu().tolist()
				gt_scores = gt_scores + label.cpu().tolist()

				loss_qa = self.l1_loss(pred.squeeze(), label.float().detach())
				loss_qa2 = self.l1_loss(pred2.squeeze(), label.float().detach())
				# =============================================================================
				# =============================================================================

				indexlabel = torch.argsort(label) # small--> large
				anchor1 = torch.unsqueeze(pred[indexlabel[0],...].contiguous(),dim=0) # d_min
				positive1 = torch.unsqueeze(pred[indexlabel[1],...].contiguous(),dim=0) # d'_min+
				negative1_1 = torch.unsqueeze(pred[indexlabel[-1],...].contiguous(),dim=0) # d_max+

				anchor2 = torch.unsqueeze(pred[indexlabel[-1],...].contiguous(),dim=0)# d_max
				positive2 = torch.unsqueeze(pred[indexlabel[-2],...].contiguous(),dim=0)# d'_max+
				negative2_1 = torch.unsqueeze(pred[indexlabel[0],...].contiguous(),dim=0)# d_min+

				# =============================================================================
				# =============================================================================

				fanchor1 = torch.unsqueeze(pred2[indexlabel[0],...].contiguous(),dim=0)
				fpositive1 = torch.unsqueeze(pred2[indexlabel[1],...].contiguous(),dim=0)
				fnegative1_1 = torch.unsqueeze(pred2[indexlabel[-1],...].contiguous(),dim=0)

				fanchor2 = torch.unsqueeze(pred2[indexlabel[-1],...].contiguous(),dim=0)
				fpositive2 = torch.unsqueeze(pred2[indexlabel[-2],...].contiguous(),dim=0)
				fnegative2_1 = torch.unsqueeze(pred2[indexlabel[0],...].contiguous(),dim=0)



				consistency = nn.L1Loss()
				assert (label[indexlabel[-1]]-label[indexlabel[1]])>=0
				assert (label[indexlabel[-2]]-label[indexlabel[0]])>=0
				triplet_loss1 = nn.TripletMarginLoss(margin=(label[indexlabel[-1]]-label[indexlabel[1]]), p=1) # d_min,d'_min,d_max
				# triplet_loss2 = nn.TripletMarginLoss(margin=label[indexlabel[0]], p=1)
				triplet_loss2 = nn.TripletMarginLoss(margin=(label[indexlabel[-2]]-label[indexlabel[0]]), p=1)
				# triplet_loss1 = nn.TripletMarginLoss(margin=label[indexlabel[-1]], p=1)
				# triplet_loss2 = nn.TripletMarginLoss(margin=label[indexlabel[0]], p=1)
				tripletlosses = triplet_loss1(anchor1, positive1, negative1_1) + \
					triplet_loss2(anchor2, positive2, negative2_1)
				ftripletlosses = triplet_loss1(fanchor1, fpositive1, fnegative1_1) + \
					triplet_loss2(fanchor2, fpositive2, fnegative2_1)
				
				loss = loss_qa + closs + loss_qa2 + closs2 + 0.5*( self.l1_loss(tripletlosses,ftripletlosses.detach())+ self.l1_loss(ftripletlosses,tripletlosses.detach()))+0.05*(tripletlosses+ftripletlosses)

				
				epoch_loss.append(loss.item())
				loss.backward()
				self.solver.step()
				

			
			
	
			modelPath = svPath + '/model_{}_{}_{}'.format(str(self.config.vesion),str(seed),epochnum)
			torch.save(self.net.state_dict(), modelPath)

			train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

			test_srcc, test_plcc = self.test(self.test_data,epochnum,svPath,seed)


			results[epochnum]=(test_srcc, test_plcc)
			with open(performPath, "r+") as file:
				data = json.load(file)
				data.update(results)
				file.seek(0)
				json.dump(data, file)
			

		
			if test_srcc > best_srcc:
				modelPathbest = svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion),str(seed))
				
				torch.save(self.net.state_dict(), modelPathbest)

				best_srcc = test_srcc
				best_plcc = test_plcc

			print('{}\t{:4.3f}\t\t{:4.4f}\t\t{:4.4f}\t\t{:4.3f}\t\t{}\t\t{:4.3f}'.format(epochnum + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc,self.paras[0]['lr'],self.droplr ))




			if (epochnum+1)==self.droplr or (epochnum+1)==(2*self.droplr) or (epochnum+1)==(3*self.droplr):

				self.lr = self.lr /self.lrratio
				
				self.paras = [{'params': self.net.parameters(), 'lr': self.lr} ]
				
				self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

		print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

		return best_srcc, best_plcc

	def test(self, data,epochnum,svPath,seed,pretrained=0):
		if pretrained:
			self.net.load_state_dict(torch.load(svPath+'/bestmodel_{}_{}'.format(str(self.config.vesion),str(seed))))
		self.net.eval()
		pred_scores = []
		gt_scores = []
		
		pbartest = tqdm(data, leave=False)

		with torch.no_grad():
			steps2 = 0
		
			
	
			for img, label in pbartest:
				img = torch.as_tensor(img.to(self.device))
				label = torch.as_tensor(label.to(self.device))
				pred,_ = self.net(img)
				
	
				pred_scores = pred_scores + pred.cpu().tolist()
				gt_scores = gt_scores + label.cpu().tolist()
				
				steps2 += 1
				
		
		
			
		pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
		gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
		

# 		if not pretrained:
		dataPath = svPath + '/test_prediction_gt_{}_{}_{}.csv'.format(str(self.config.vesion),str(seed),epochnum)
		with open(dataPath, 'w') as f:
			writer = csv.writer(f)
			writer.writerows(zip(pred_scores, gt_scores))
			
			
		test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
		test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
		return test_srcc, test_plcc
	
if __name__=='__main__':
	import os
	import argparse
	import random
	import numpy as np
	from args import *
	
	