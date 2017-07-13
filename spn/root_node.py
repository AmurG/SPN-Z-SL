import copy
import numpy as np
from .node import Node

#Mostly wrapper functions here. You may ignore this file - note that MPE is basically a maximum inference procedure 
#Hard EM is a particular case of EM where the assignment of a point to a cluster is done fully and not partially. Suppose in an EM iteration, a point belongs to cluster A with p = 0.4 and to B with 1-p = 0.6, then hard EM does the next cluster with the point in B whereas soft EM would proceed by allocating 0.4 of p to A and 0.6 of it to B.

class RootNode(Node):
	def __init__(self, node):
		super(RootNode, self).__init__(node.n, node.scope)
		self.children.append(node)
		node.parent = self

	def evaluate(self, obs, mpe=False):
		return self.children[0].evaluate(obs, mpe=mpe)

	def update(self, obs, params):
		self.children[0].update(obs, params)
		self.n += len(obs)

	def display(self, depth=0):
		self.children[0].display(depth)

	def add_child(self, child):
		assert len(self.children) == 0
		self.children.append(child)
		child.parent = self

	def remove_child(self, child):
		assert len(self.children) == 1
		self.children.remove(child)
		child.parent = None

	def check_valid(self):
		assert self.children[0].check_valid()

	def hard_em(self, data):
		inds = np.arange(len(data))
		self.children[0].hard_em(data, inds)

	def normalize_nodes(self):
		self.children[0].normalize_nodes()
