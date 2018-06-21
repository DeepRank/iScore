import sys, os
import numpy as np

from iScore.graph import Graph
import pickle
import tarfile

from svmutil import *


class DataSet(object):

	def __init__(self,trainID,Kfile,maxlen,testID=None):

		print(trainID)
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
	def _get_ids(idlist):

		if isinstance(idlist,str):
			if os.path.isfile(idlist):

				with open(idlist,'r') as  f:
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
			else:
				raise FileNotFoundError(idlist, 'is not a file')

		elif isinstance(idlist,list):

			nl = len(idlist)
			nc = len(idlist[0])
			if nc == 2:
				classes = [id_[0] for id_ in idlist]
				names = [id_[1] for id_ in idlist]
				return names, classes
			else:
				names = idlist
				classes = [0]*nl
				return names, classes

		else:
			raise ValueError(idlist, 'not a proper IDs file')

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
		key = list(K.keys())[1]
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

	def predict(self,package_name):

		if self.testDataSet is None:
			raise ValueError('You should specify a testDataSet')

		# extrat the model
		tar = tarfile.open(package_name)
		dict_tar = dict(zip(tar.getnames(),tar.getmembers()))
		tar.makefile(dict_tar['svm_model.pkl'],'./_tmp_model.pkl')
		model = svm_load_model('./_tmp_model.pkl')

		# pedict the classes
		self.testDataSet.iScore = svm_predict(self.testDataSet.test_class,self.testDataSet.Kmat,model)

		# clean uo crew
		os.remove('./_tmp_model.pkl')


def iscore_svm(train=False,train_class='caseID.lst',trainID=None,testID=None,
				kernel='./kernel/',save_model='svm_model.pkl',load_model=None,
				package_model=False,package_name=None,graph='./graph/',
				include_kernel=False, maxlen = None):

	# figure out the kernel files
	# if a dir was given all the file in that dir are considered
	if os.path.isdir(kernel):
		Kfile =  [kernel + f for f in os.listdir(kernel)]
	elif os.path.isfile(kernel):
		Kfile = kernel
	else:
		raise ValueError('Kernel file not found')

	# train the model
	if train:

		traindata = DataSet(train_class,Kfile,maxlen)
		svm = SVM(trainDataSet=traindata)
		svm.train(model_file_name=save_model)

		if package_model:
			print('Create Archive file : ', package_name)
			svm.archive(graph_path=graph,
				        kernel_path=kernel,
				        include_kernel=include_kernel,
				        model_name=package_name)

	else:

		if trainID is None:
			tar = tarfile.open(package_name)
			members = tar.getmembers()
			trainID = [os.path.splitext(os.path.basename(m.name))[0] for m in members if m.name.startswith('./graph/')]

		if testID is None:
			testID = [os.path.splitext(n)[0] for n in os.listdir('./graph/')]

		testdata = DataSet(trainID,Kfile,maxlen,testID=testID)
		svm = SVM(testDataSet = testdata)
		svm.predict(package_name = package_name)