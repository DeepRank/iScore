import sys, os
import numpy as np

from iScore.graphrank.graph import Graph
import pickle
import tarfile

from libsvm.svmutil import *


class DataSet(object):

    def __init__(self,trainID,Kfile,maxlen,testID=None):
        """Cretae a data set for SVM.

        Args:
            trainID (str or list): comformation ID for the training set. if str it should be the dir where the graphs are stored
            Kfile (TYPE): File name containig th K matrix
            maxlen (TYPE): maximu wlak length for the kernel calculations
            testID (None, optional): comformation ID for the test set. if str it should be the dir where the graphs are stored
        """
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
        """Get the ID if the training/test set

        Args:
            idlist (str or list): if str should be a file containg : name class

        Returns:
            list(str),list(int): names anc ground truth of the set

        Raises:
            FileNotFoundError: If idlist is not an existing file
            ValueError: If the format of idlist is not understood
        """
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
                    print('Warning: Ground truth classes not found in '+idlist)
                    names = data[:,0]
                    classes = [None]*nl
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
                #print('Warning: Ground truth classes not found in list' )
                names = idlist
                classes = [None]*nl
                return names, classes

        else:
            raise ValueError(idlist, 'not a proper IDs file')

    def get_K_matrix(self):
        """get the Kernel matrix of the set

        Raises:
            ValueError: if maxlen was specified and is larger than the maximum possible length
        """
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

                    print('Warning : Graph combination (%s,%s) not found in files' %(name_test,name_train))
                    for f in self.Kfile:
                        print('\t\t',f)
                    #raise ValueError('Graphs not Found')

        self.Kmat = self.Kmat.tolist()

class SVM(object):

    def __init__(self,trainDataSet=None,testDataSet=None,load_model=None):
        """Class that handles the SVM training/testing process

        Args:
            trainDataSet (None, DataSet, optional): data set for training
            testDataSet (None, DataSet, optional): data set for testing
            load_model (None, optional): load a model
        """
        self.trainDataSet = trainDataSet
        self.testDataSet = testDataSet
        if load_model is not None:
            self.model = svm_load_model(load_model)

    def train(self,model_file_name=None):
        """Train the model using the the train dataset

        Args:
            model_file_name (None, str, optional): File name to save the model

        Raises:
            ValueError: If no training data set was specified
        """
        if self.trainDataSet is None:
            raise ValueError('You should specify a trainDataSet')

        print('Training Model')
        kdata = []
        for i,k in enumerate(self.trainDataSet.Kmat):
            kdata.append([i+1]+k)

        prob = svm_problem(self.trainDataSet.train_class,kdata,isKernel=True)
        #param = svm_parameter('-t 4 -c 4 -b 1')
        param = svm_parameter('-t 4')
        self.model = svm_train(prob,param)
        self.mode_file_name = model_file_name

        if model_file_name is not None:
            svm_save_model(model=self.model,model_file_name=model_file_name)

    def archive(self,graph_path='./graph/',kernel_path='./kernel/',
                include_kernel=False,model_name='training_set.tar.gz'):
        """Create an archive file to store the model and the graphs/kernels

        Args:
            graph_path (str, optional): directory containing the graphs
            kernel_path (str, optional): directory containing the kernels
            include_kernel (bool, optional): include the kernel file in the archive
            model_name (str, optional): file name of the archive

        Raises:
            ValueError: if the grapg or kernel dir do not exists
        """
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
        """Predict the class of a test set.

        Args:
            package_name (str): archive file containg the model

        Raises:
            ValueError: if no test set are specified
        """
        if self.testDataSet is None:
            raise ValueError('You should specify a testDataSet')

        # extrat the model
        tar = tarfile.open(package_name)
        dict_tar = dict(zip(tar.getnames(),tar.getmembers()))
        tar.makefile(dict_tar['svm_model.pkl'],'./_tmp_model.pkl')
        model = svm_load_model('./_tmp_model.pkl')

        # check if we have ground truth
        ground_truth = self.testDataSet.test_class
        if None in ground_truth:
            print('Warning ground truth missing. SVM accuracy will be meaningless')
            ground_truth = [0]*len(ground_truth)

        # pedict the classes
        self.testDataSet.iScore = svm_predict(ground_truth,self.testDataSet.Kmat,model)
        #print(self.testDataSet.iScore)

        # clean up crew
        os.remove('./_tmp_model.pkl')

    def export_prediction(self,fname):
        """Export the predicted values to file and/or pickle it

        Args:
            fname (str): file name
        """
        if fname.endswith('.pkl') or fname.endswith('.pckl'):
            self._export_score_pickle(fname,
                                      self.testDataSet.test_name,
                                      self.testDataSet.test_class,
                                      self.testDataSet.iScore)


        elif fname.endswith('.dat') or fname.endswith('.txt'):
            self._export_score_text(fname,
                                    self.testDataSet.test_name,
                                    self.testDataSet.test_class,
                                    self.testDataSet.iScore)

        else:
            fname = os.path.splitext(fname)[0]
            self._export_score_pickle(fname+'.pkl',
                          self.testDataSet.test_name,
                          self.testDataSet.test_class,
                          self.testDataSet.iScore)
            self._export_score_text(fname+'.dat',
                                    self.testDataSet.test_name,
                                    self.testDataSet.test_class,
                                    self.testDataSet.iScore)


    @staticmethod
    def _export_score_pickle(fname,name,label,score):
        """Export the prediction as pickle file

        Args:
            fname (str): file name
            name (list(str)): list of the conformation names
            label (TYPE): list of the conformation ground truth label
            score (TYPE): iScore
        """
        data = {}
        for i,n in enumerate(name):
            data[n] = {'ground_truth' : label[i],
                       'prediction' : score[0][i],
                       'decision_value' : score[2][i]}
        f = open(fname,'wb')
        pickle.dump(data,f)
        f.close()

    @staticmethod
    def _export_score_text(fname,name,label,score):
        """Export the prediction as text file

        Args:
            fname (str): file name
            name (list(str)): list of the conformation names
            label (TYPE): list of the conformation ground truth label
            score (TYPE): iScore
        """
        f = open(fname,'w')
        f.write('{:10} {:>5}     {:>5}     {:>14}\n'.format('#Name','label','pred','decision_value'))
        for i,n in enumerate(name):
            if label[i] is None:
                st = "{:10} {:>5}     {:5}     {: 14.3f}\n"
                il = 'None'
            else:
                st = "{:10} {:5}     {:5d}     {: 14.3f}\n"
                il = int(label[i])
            f.write(st.format(n,il,int(score[0][i]),score[2][i][0]))
        f.close()

def iscore_svm(train=False,train_class='caseID.lst',trainID=None,testID=None,
                kernel='./kernel/',save_model='svm_model.pkl',load_model=None,
                package_model=False,package_name=None,graph='./graph/',
                include_kernel=False, maxlen = None,score_file='GraphRank'):
    """Function called in the binary iScore.predict and iScore.train

    Args:
        train (bool, optional): train or predict
        train_class (str, optional): file name containing the ID and classes of the train set
        trainID (None, optional): file containing the ID of the train set
        testID (None, optional): file containing the ID of the test set
        kernel (str, optional): directory containing the kernel files
        save_model (str, optional): save the model in a pickle file after training
        load_model (None, optional): load a model for testing
        package_model (bool, optional): Create an archive file containing the training set
        package_name (None, optional): Name of the archive file
        graph (str, optional): directory containing the graphs
        include_kernel (bool, optional): Include the kernels in the archive file
        maxlen (None, optional): maximum walk length
        score_file (str, optional): output file containg the prediction

    Raises:
        ValueError: If the kernel files are nout found
    """

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

    # use a trained model for prediction
    else:

        if trainID is None:
            tar = tarfile.open(package_name)
            members = tar.getmembers()
            trainID = [os.path.splitext(os.path.basename(m.name))[0] for m in members if '/graph/' in m.name]

        if testID is None:
            testID = [os.path.splitext(n)[0] for n in os.listdir('./graph/')]
        else:
            if os.path.isdir(testID):
                testID = [os.path.splitext(n)[0] for n in os.listdir(testID)]
            elif os.path.isfile(testID):
                testID = testID

        testdata = DataSet(trainID,Kfile,maxlen,testID=testID)
        svm = SVM(testDataSet = testdata)
        svm.predict(package_name = package_name)
        svm.export_prediction(score_file)