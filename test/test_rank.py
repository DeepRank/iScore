import os
from iScore.graphrank.graph import iscore_graph
from iScore.graphrank.kernel import iscore_kernel
from iScore.graphrank.rank import iscore_svm
import unittest

class TestRank(unittest.TestCase):
    """Test the pipeline."""

    def setUp(self):

        self.train_pdb = './rank/train/pdb/'
        self.train_pssm = './rank/train/pssm/'
        self.trainID = './rank/train/caseID.lst'
        self.train_graph = './rank/train/graph/'
        self.train_kernel = './rank/train/kernel/'
        self.train_archive = 'training_set.tar.gz'

        self.test_pdb = './rank/test/pdb/'
        self.test_pssm = './rank/test/pssm/'
        self.test_graph = './rank/test/graph/'
        self.test_kernel = './rank/test/kernel/'

    def test1_svmtrain(self):

        # create graphs
        if not os.path.isdir(self.train_graph):
            os.mkdir(self.train_graph)
        iscore_graph(pdb_path=self.train_pdb,
                     pssm_path=self.train_pssm,
                     outdir=self.train_graph)

        # kernel
        if  not os.path.isdir(self.train_kernel):
            os.mkdir(self.train_kernel)
        kfile = os.path.join(self.train_kernel,'kernel.pckl')
        iscore_kernel(train_graph=self.train_graph,
                      test_graph=self.train_graph,
                      outfile=kfile)

        # train the model
        iscore_svm(train=True,
                   train_class=self.trainID,
                   kernel=self.train_kernel,
                   graph=self.train_graph,
                   package_model=True,
                   package_name=self.train_archive)

    def test2_svmtest(self):

        # create the graphs
        if not os.path.isdir(self.test_graph):
            os.mkdir(self.test_graph)
        iscore_graph(pdb_path=self.test_pdb,
                     pssm_path=self.test_pssm,
                     outdir=self.test_graph)

        # compute the kernels
        if not os.path.isdir(self.test_kernel):
            os.mkdir(self.test_kernel)
        kfile = os.path.join(self.test_kernel,'kernel.pckl')
        iscore_kernel(test_graph=self.test_graph,
                      train_archive=self.train_archive,
                      outfile=kfile)

        # predcit the classes
        iscore_svm(kernel=self.test_kernel,
                   testID=self.test_graph,
                   graph=self.test_graph,
                   package_name=self.train_archive)