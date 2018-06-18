import sys, os
import numpy as np

from iScore.graph import Graph
import pickle
import tarfile

pathsvm = '/home/nico/programs/libsvm/python'
sys.path.append(pathsvm)
from svmutil import *


class DataSet(object):

	def __init__(self,trainID,Kfile,maxlen,testID=None):


		self.train_name, self.train_class = self._get_ids(trainID)

		if testID is None:
			self.test_name = self.train_name
			self.test_class = self.train_class
		else:
			self.test_name, self.test_class = self._get_ids(testID)

		self.Kfile = Kfile

		self.ntrain = len(self.train_name)
		self.ntest = len(self.test_name)

		self.Kmat = np.zeros((self.ntest,self.ntrain))

		self.maxlen = maxlen

		self.get_K_matrix()

	@staticmethod
	def _get_ids(fname):
		with open(fname,'r') as  f:
			data = f.readlines()
		data = np.array([l.split() for l in data])
		nl,nc = data.shape

		if nc == 2:
			classes = data[:,0].astype('int')
			names = data[:,1]
			return names.tolist(), classes.tolist()
		else:
			names = data[:,0]
			classes = [0]*nl
			return names.tolist(), classes

	def get_K_matrix(self):

		if not isinstance(self.Kfile,list):
			self.Kfile = [self.Kfile]

		K = dict()
		for f in self.Kfile:
			K.update(pickle.load(open(f,'rb')))

		# get the max walk len
		#max_possible_len = K['param']['walk']

		# see if we can add extension (obsolete)
		addext = False
		key = list(K.keys())[4]
		max_possible_len = len(K[key])
		if key[0].endswith('.pckl'):
			addext = True

		# get the max walk len
		if self.maxlen is None:
			self.maxlen = max_possible_len

		# chek if that's ok
		elif self.maxlen > max_possible_len:
			print('Error : Maximum walk length possible in kernel file : %d ' %max_possible_len)
			print('      : Requested walk length                       : %d ' %maxlen)
			raise ValueError('maxlen too large')

		for itest,name_test in enumerate(self.test_name):

			if addext:
				name_test += '.pckl'

			for itrain,name_train in enumerate(self.train_name):

				if addext:
					name_train += '.pckl'

				if (name_test,name_train) in K:
					self.Kmat[itest,itrain] = np.sum(K[(name_test,name_train)][:self.maxlen])
				elif (name_train,name_test) in K:
					self.Kmat[itest,itrain] = np.sum(K[(name_train,name_test)][:self.maxlen])
				else:
					print('Error : Graph combination (%s,%s) not found in files' %(name_test,name_train))
					for f in self.Kfile:
						print('\t\t',f)
					raise ValueError('Graphs not Found')

		self.Kmat = self.Kmat.tolist()

class SVM(object):

	def __init__(self,trainDataSet=None,testDataSet=None,load_model=None):
		self.trainDataSet = trainDataSet
		self.testDataSet = testDataSet
		if load_model is not None:
			self.model = svm_load_model(load_model)

	def train(self,model_file_name=None):

		if self.trainDataSet is None:
			raise ValueError('You should specify a trainDataSet')

		print('Training Model')
		prob = svm_problem(self.trainDataSet.train_class,self.trainDataSet.Kmat)
		param = svm_parameter('-c 4')
		self.model = svm_train(prob,param)
		self.mode_file_name = model_file_name

		if model_file_name is not None:
			svm_save_model(model=self.model,model_file_name=model_file_name)

	def archive(self,graph_path='./graph/',kernel_path='./kernel/',
		        include_kernel=False,model_name='training_set.tar.gz'):

		# init the model
		tar = tarfile.open(model_name,"w:gz")

		# get the graphs
		if not os.path.isdir(graph_path):
			raise ValueError('Graph directory %s does not exist' %graph_path)
		graph_names = os.listdir(graph_path)


		for g in graph_names:
			gfile = os.path.join(graph_path,g)
			tar.add(gfile)

		#get the kernel
		if include_kernel:

			if not os.path.isdir(kernel_path):
				raise ValueError('Kernel directory %s does not exist' )

			kernel_names = os.listdir(kernel_path)

			for k in kernel_names:
				tar.add(kfile)

		# get the svm_model
		tar.add(self.mode_file_name)
		tar.close()

	def predict(self):

		if self.testDataSet is None:
			raise ValueError('You should specify a testDataSet')

		self.testDataSet.iScore = svm_predict(self.testDataSet.test_class,self.testDataSet.Kmat,self.model)



if __name__ == '__main__':

	trainID = '../training_set/caseID.lst'
	path = '../training_set/kernel/'
	Kfile =  [path+f for f in os.listdir(path)]
	maxlen = 4

	traindata = DataSet(trainID,Kfile,maxlen)
	svm = SVM(trainDataSet=traindata)
	svm.train()