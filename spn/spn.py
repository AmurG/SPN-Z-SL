import numpy as np

from .root_node import RootNode
from .sum_node import SumNode
from .product_node import ProductNode
from .normal_leaf_node import NormalLeafNode
from .multi_normal_leaf_node import MultiNormalLeafNode

#Code is straightforward : Alternately k-mean cluster and add sum nodes and on the other half, induce product nodes. Corrthresh is the most important hyperparameter for tuning in effect. Equalweight corresponds to a bayesian interpretation vs frequentist. Consider two clusters in the data constituting 10% and 90% of the data. So in this case the SPN forms a sum node with two children with weights 0.1 - cluster A - and 0.9 for cluster B. Now a new sample comes along, and the likelihood from cluster A is > from cluster B. However, A has a lower prior probability, so it's possible that P(sample|A)P(A) < P(sample|B)P(B) even though P(sample|A) > P(sample|B).

#The mvscope parameter should not be too large, but neither too small. A large mvscope moves the model closer and closer to GMM performance. One good sanity check is to do this and see if the average Log-likelihood from the SPN is moving closer or further from GMM performance, if further, there's likely some issue. ( Test on abalone or Iris for this - they are standard benchmarks for this kind of thing )

class SPNParams:
	"""
	Parameters
	----------
	batchsize : number of samples in a mini-batch.
	            if 0, use the entire set as one batch.
	mergebatch : number of samples a product node needs to see before updating
	             its structure.
	corrthresh : correlation coefficient threshold above which two variables
	             are considered correlated.
	equalweight : whether sum nodes should consider children as having equal
	              weights when deciding which children to pass data to.
	updatestruct : whether to update the network structure.
	mvmaxscope : number of variables that can be combined into a multivariate
	             leaf node.
	leaftype : type of leaf nodes, one of "normal", "binary", "binarynormal".
	"""
	def __init__(self, batchsize=128, mergebatch=128, corrthresh=0.1,
	             equalweight=True, updatestruct=True,
	             mvmaxscope=2, leaftype="normal"):
		if leaftype != "normal":
			raise ValueError("Leaf type {0} not supported.".format(leaftype))
		self.batchsize = batchsize
		self.mergebatch = mergebatch
		self.corrthresh = corrthresh
		self.equalweight = equalweight
		self.updatestruct = updatestruct
		self.mvmaxscope = mvmaxscope
		self.leaftype = leaftype
		self.binary = False if leaftype=="normal" else True

class SPN:
	"""
	Parameters
	----------
	node : int or Node
		if int, number of variables
		if Node, root of network
	params : SPNParams
		parameters of the network
	"""
	def __init__(self, node, numcomp, params):
		if type(node) == int:
			numvar = node
			scope = np.arange(numvar)
#			node = make_product_net(scope, params.leaftype)
			node = init_root(scope, numcomp, params.leaftype)
		self.root = RootNode(node)
		self.params = params

	def evaluate(self, obs):
		if obs.ndim == 1:
			obs = obs.reshape(1, len(obs))
		return self.root.evaluate(obs)

	def update(self, obs, upd):
		if upd:
#			self.params.corrthresh = 20.2
			i = 100
			a = 0
			b = a + i
			while (a != len(obs)):
				self.root.update(obs[a:b], self.params)
				i *= 2
				a = b
				b = min(len(obs), b+i)
				print(a)
			return
#		self.params.corrthresh = 0.2;
		if obs.ndim == 1:
			obs = obs.reshape(1, len(obs))
		if self.params.batchsize > 0:
			for i in range(0, len(obs), self.params.batchsize):
				print(i)
				self.root.update(obs[i:i+self.params.batchsize], self.params)
				# if i % 1000 == 0:
				# 	self.normalize_nodes()
		else:
			self.root.update(obs, self.params)

	def display(self):
		self.root.display()

	def normalize_nodes(self):
		self.root.normalize_nodes()

def init_root(scope, nc, leaftype):
	node = SumNode(0, scope)
	children = [make_product_net(scope, leaftype) for i in range(nc)]
	node.add_children(*children)
	return node

def make_product_net(scope, leaftype):
	node = ProductNode(0, scope, leaftype)
	for v in scope:
		node.add_child(node.Leaf(0, v))
	return node

