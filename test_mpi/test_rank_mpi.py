import os
from iScore.graphrank.graph import iscore_graph_mpi
from iScore.graphrank.kernel import iscore_kernel_mpi

import unittest
from mpi4py import MPI

class TestMPI(unittest.TestCase):
    """Test the pipeline."""

    def setUp(self):

        self.train_pdb = './rank/pdb/'
        self.train_pssm = './rank/pssm/'
        self.train_graph = './rank/graph/'
        self.train_kernel = './rank/kernel/'
        self.train_archive = 'training_set.tar.gz'

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def test1_graph(self):

        # create graphs
        if not os.path.isdir(self.train_graph):
            os.mkdir(self.train_graph)
        iscore_graph_mpi(pdb_path=self.train_pdb,
                         pssm_path=self.train_pssm,
                         outdir=self.train_graph,
                         rank=self.rank,
                         size=self.size,
                         mpi_comm=self.comm);


    def test2_kernel(self):

        # kernel
        if  not os.path.isdir(self.train_kernel):
            os.mkdir(self.train_kernel)
        kfile = os.path.join(self.train_kernel,'kernel.pckl')
        iscore_kernel_mpi(train_graph=self.train_graph,
                      test_graph=self.train_graph,
                      outfile=kfile,rank=self.rank,size=self.size)