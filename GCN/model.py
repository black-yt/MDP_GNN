import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GraphConvolution(nn.Module):
	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).to(device))
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features).to(device))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)


	def forward(self, x, adj):
		support = torch.mm(x.to(device), self.weight.to(device))
		output = torch.mm(adj.to(device), support.to(device))
		if self.bias is not None:
			return output + self.bias
		else:
			return output


	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'


class GCN(nn.Module):
	def __init__(self, in_features, nhid, nclass, dropout):
		super(GCN, self).__init__()
		self.in_features = in_features
		self.nhid = nhid
		self.nclass = nclass
		self.dropout = dropout
		self.gcn1 = GraphConvolution(in_features, nhid)
		self.gcn2 = GraphConvolution(nhid, nclass)


	def forward(self, x, adj):
		h1 = F.relu(self.gcn1(x, adj))
		h1_d = F.dropout(h1, self.dropout, training=self.training)
		logits = self.gcn2(h1_d, adj)
		output = F.log_softmax(logits, dim=1)
		return output
