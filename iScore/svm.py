import sys
from .gaph import Graph
import pickle

pathsvm = '/home/nico/programs/libsvm/python'
sys.path.append(pathsvm)
from svm import *


class DataSet(object):

	def __init__(self,testID=None,trainID,Kfile,maxlen):


		self.train_name, self.train_class = self._get_ids(trainID)
		if testID is not None:
			self.test_name, self.test_class = self._get_ids(testID)
		else:
			self.test_name = self.train_name
			self.test_class = self.train_class

		self.Kfile = Kfile

		self.ntrain = len(self.train_name)
		self.ntest = len(self.test_name)

		self.Kmat = np.zeros(self.ntest,self.ntrain)


	@staticmethod
	def _get_ids(fname):
		with open(fname,'r') as  f:
			data = f.readlines()
		data = np.array([l.split() for l in data])
		names = data[:,1]
		classes = data[:,0]
		return names, classes

	def get_K_matrix(self):

		K = pickle.load(open(self.Kfile,'rb'))

		for itest,name_test in enumerate(self.test_name):
			for itrain,name_train in enumerate(self.train_name):
				if (name_test,name_train) in K:
					self.Kmat[itest,itrain] = np.sum(K[(name_test,name_train)][:maxlen])
				elif (name_train,name_test) in K:
					self.Kmat[itest,itrain] = np.sum(K[(name_train,name_test)][:maxlen])
				else:
					raise ValueError('Graph combination (%s,%s) not found in file %s' %(name_test,name_train,self.Kfile))

class SVM(object):

	def __init__(self,trainDataSet=None,testDataSet=None,load_model=None):
		self.trainDataSet = None
		self.testDataSet = None
		self.load_model = None

	def train(self,trainDataSet=None,model='svm_model.pckl'):

		if trainDataSet is not None:
			self.trainDataSet = trainDataSet
		if self.trainDataSet is None:
			raise ValueError('You should specify a trainDataSet')

		prob = svm_problem(self.trainDataSet.train_class,self.trainDataSet.Kmat)
		param = svm_parameter('-c 4')
		model = libsvm.svm_train(prob,param)
		svm_save_model(model)

	def predict(self,testDataSet=None,model='iScore_SVM_Model.pckl'):

		if testDataSet is not None:
			self.testDataSet = testDataSet
		if self.testDataSet is None:
			raise ValueError('You should specify a testDataSet')

		model = svm_load_model(model)
		K, max_idx = gen_svm_nodearray(self.testDataSet.Kmat,isKernel=True)
		self.testDataSet.iScore = libsvm.svm_predict(model,K)