#!/usr/bin/env python

import scipy.io as spio
import scipy.sparse as sp_sparse
import numpy as np
from time import time
import itertools
from collections import OrderedDict
import pickle
import os
import tarfile

from .graph import Graph

try:
    import pycuda.autoinit
    from pycuda import driver, compiler, gpuarray, tools
    from pycuda.reduction import ReductionKernel

except ModuleNotFoundError:
    print('Warning : pycuda not found')

try:
    import skcuda.linalg as culinalg
    import skcuda.misc as cumisc
    culinalg.init()
except ModuleNotFoundError:
    print('Warning : scikit-cuda not found')



class Kernel(object):

    def __init__(self,testIDs='testID.lst',trainIDs='trainID.lst',
                 test_graph='./graph/',train_graph='./graph/',
                 train_archive=None, debug=False,
                 gpu_block=(8,8,1),method='vect'):

        """Compute the kernels of graph pairs.
        Main class to compute the graph similarities.

        Example:

        >>>from iScore.graph import Graph
        >>>from iScore.kernel import Kernel
        >>># init and load the data
        >>>ker = Kernel()
        >>>ker.import_from_mat()
        >>># get the path of the check file
        >>>checkfile = ker.get_check_file()
        >>># run the calculations
        >>>ker.run(lamb=1.0,walk=4,check=checkfile)


        Args:
            testIDs (str, optional): name of the file contaings the name of the test conformations
            trainIDs (str, optional): name of the file contaings the name of the train conformations
            test_graph (str, optional): Path to the directory containing the test graphs
            train_graph (str, optional): Path to the directory containing the train graphs
            gpu_block (tuple, optional): Block size to use for GPU
            method (str, optional): Method to compute the graph similarities. Options are iter, combvec and vect (default vect is much faster)
        """

        # all the options
        self.trainIDs = trainIDs
        self.testIDs = testIDs
        self.train_graph = train_graph
        self.test_graph = test_graph
        self.gpu_block = gpu_block
        self.method = method
        self.train_archive=train_archive
        self.debug=debug

        # the cuda kernel
        self.kernel = os.path.dirname(os.path.abspath(__file__)) + '/cuda/cuda_kernel.cu'

        # check if the kernel exists
        if not os.path.isfile(self.kernel):
            raise FileNotFoundError('Cuda kernel %s not found' %self.kernel)


    ##############################################################
    #
    # Import the data from the precomputed Matlab/Pickle file
    #
    ##############################################################

    def import_from_mat(self,mpi_rank=0,mpi_size=1):
        """Import the data from the selected graph files."""

        self.max_num_edges_train = 0
        self.max_num_edges_test = 0

        if self.trainIDs is not None:
            if not os.path.isfile(self.trainIDs):
                raise FileNotFoundError('file %s not found' %(self.trainIDs))

        if self.testIDs is not None:
            if not os.path.isfile(self.testIDs):
                raise FileNotFoundError('file %s not found' %(self.testIDs))

        if not os.path.isdir(self.test_graph):
            raise NotADirectoryError('Directory %s not found' %(self.test_graph))

        if not os.path.isdir(self.train_graph):
            raise NotADirectoryError('Directory %s not found' %(self.train_graph))

        if self.train_archive is not None:
            if not os.path.isfile(self.train_archive):
                raise FileNotFoundError('file %s not found' %(self.train_archive))

        # get the names of the data
        # Warning those are not in the same order
        # than  in the file
        # bloody filter !!
        self.train_graphs = OrderedDict()
        if self.train_archive is not None:
            tar=tarfile.open(self.train_archive)
            for member in tar.getmembers():
                if '/graph/' in member.name:
                    name = member.name.split('/')[-1]
                    self.train_graphs[name] = pickle.load(tar.extractfile(member))

        else:
            train_names = self._get_file_names(self.trainIDs,self.train_graph)
            for name in train_names:
                g = Graph(self.train_graph + '/' + name)
                if g.num_nodes > 0:
                    self.train_graphs[name] = g
                    self.max_num_edges_train = np.max([self.max_num_edges_train,self.train_graphs[name].num_edges])
                else:
                    print('Warning : Graph %s was exclued from the training data (num_ndodes = 0)' %name)

        test_names  = self._get_file_names(self.testIDs,self.test_graph)

        # split the test names
        if mpi_size > 1:
            nitem = int(np.ceil(len(test_names)/mpi_size))
            start, end = mpi_rank*nitem, (mpi_rank+1)*nitem
            test_names = test_names[start:end]

        # get the test graphs
        self.test_graphs = OrderedDict()
        for name in test_names:
            g = Graph(self.test_graph + '/' + name)
            if g.num_nodes > 0:
                self.test_graphs[name] = Graph(self.test_graph + '/' + name)
                self.max_num_edges_test = np.max([self.max_num_edges_test,self.test_graphs[name].num_edges])
            else:
                print('Warning : Graph %s was exclued from the testing data (num_ndodes = 0)' %name)

    @staticmethod
    def _get_file_names(filename,dirname):
        """Get the graph file names

        Args:
            filename (str): name of the file containing the mol names
            dirname (TYPE): path to dir containing the graphs

        Returns:
            list(str): List containing the graph names
        """

        # get the file name in the graph dir
        existing_names = os.listdir(dirname)

        if filename is None:
            return existing_names

        else:

            # get the requred complex names
            with open(filename) as f:
                required_names = tuple([name.split()[0]+'.' for name in f.readlines() if name.split()])

            # get the file names
            names =  list(filter(lambda x: x.startswith(required_names),existing_names))

            if len(names) != len(required_names):
                Warning('Not all required graphs were found')

            return names

    def get_check_file(self,fname=None):

        """Get the name of the check file that hold precomputed values of K.
        Tries to find the check file.
        Returns the filename is it's not None
        Returns ./kernelMAT/K_<test_name>.mat if this file exists
        Returns None otherwise

        Args:
            fname (None, optional): Check file

        Returns:
            str or None: Checkfile name
        """
        if fname is None:
            return None

        elif os.path.isfile(fname):
            return fname

        elif fname=='auto':
            test_name = os.path.splitext(self.testIDs)[0]
            kname = 'kernelMAT/K_' + test_name + '.mat'
            if os.path.isfile(kname):
                return kname
            else:
                return None

    ##############################################################
    #
    # Main function to compute all the data
    #
    ##############################################################

    def run(self,lamb,walk,outfile='kernel.pkl',cuda=False,gpu_block=(8,8,1),check=None,
            test=False,mpi_rank=0,mpi_size=1):
        """Compute all the K values for all the graph pairs.

        Args:
            lamb (float): value for lambda
            walk (int): length of the walk
            outfile (str, optional): File where to store the K values
            cuda (bool, optional): Use CUDA or not
            gpu_block (tuple, optional): Size of the gpu block
            check (None, optional): Check the results
            test (bool,otional): if True only compute the first pair (no output saved)
        """

        if cuda and mpi_size > 1:
            print('MPI and CUDA implementation not supported (yet)\n CUDA disabled.')
            cuda = False

        # do all the single-time cuda operations
        if cuda:

            # compile the kernel
            self.compile_kernel()

            # prebook the weight and index matrix
            # required for the calculation of self.Wx
            t0 = time()
            n1 = self.max_num_edges_test
            n2 = self.max_num_edges_train
            n_edges_prod = 2*n1*n2
            self.weight_product = gpuarray.zeros(n_edges_prod, np.float32)
            self.index_product = gpuarray.zeros((n_edges_prod,2), np.int32)
            if self.debug:
                print('GPU - Mem  : %f' %(time()-t0))

        # store the parametres
        K = {}
        K['param'] = {'lambda':lamb,'walk':walk,'cuda':cuda,'gpu_block':gpu_block}

        # check file if it exists
        if check is not None:

            if not os.path.isfile(check):
                raise FileNotFoundError(check)

            data = spio.loadmat(check,squeeze_me=True)
            Kcheck = data['K']

            testID_check = data['graphIDs_test']
            if isinstance(testID_check,str):
                testID_check = [testID_check]
            else:
                testID_check = testID_check.tolist()

            trainID_check = data['graphIDs_train']
            if isinstance(trainID_check,str):
                trainID_check = [trainID_check]
            else:
                trainID_check = trainID_check.tolist()

        # rename the output file if mpi
        if mpi_size > 1:
            fname,ext = os.path.splitext(outfile)
            outfile = fname + '_{:04d}'.format(mpi_rank) + ext

        # store the check
        check_values = []

        # go through all the data
        for (name1,G1) in self.test_graphs.items():
            for (name2,G2) in self.train_graphs.items():

                print('')
                print(mpi_rank,name1,name2)
                print('-'*20)
                t0 = time()

                # compute the matrices
                if cuda:
                    self.compute_kron_mat_cuda(G1,G2)
                    self.compute_px_cuda(G1,G2)
                    self.compute_W0_cuda(G1,G2)

                else:
                    self.compute_kron_mat(G1,G2)
                    self.compute_px(G1,G2)
                    self.compute_W0(G1,G2)

                # compute the graphs
                n1 = os.path.splitext(name1)[0]
                n2 = os.path.splitext(name2)[0]
                K[(n1,n2)] = self.compute_K(lamb=lamb,walk=walk)

                if self.debug:
                    print('Total      : %f' %(time()-t0))
                    print('-'*20)
                    print('K      :  ' + '  '.join(list(map(lambda x: '{:1.3}'.format(x),K[(n1,n2)]))))

                # print the check if present
                if check is not None:

                    ind_test = testID_check.index(n1)
                    ind_train = trainID_check.index(n2)
                    if Kcheck.ndim > 1:
                        print('Kcheck :  ' + '  '.join(list(map(lambda x: '{:1.3}'.format(x),Kcheck[ind_test][ind_train]))))
                        kc = Kcheck[ind_test][ind_train]
                    else:
                        print('Kcheck :  ' + '  '.join(list(map(lambda x: '{:1.3}'.format(x),Kcheck))))
                        kc = Kcheck
                    check_values.append(np.allclose(K[(n1,n2)],kc))

                if test:
                    exit()

        # save the data
        f = open(outfile,'wb')
        pickle.dump(K,f)
        f.close()

        return check_values

    ##############################################################
    #
    # Calculation of the K (CPU only)
    #
    ##############################################################

    def compute_K(self,lamb=1,walk=4):
        """Compute the kernel

        Args:
            lamb (int, optional): value of lambda
            walk (int, optional): length of the walk

        Returns:
            list(float): values of the kernel
        """
        t0 = time()
        self.px /= np.sum(self.px)
        K = np.zeros(walk+1)
        K[0] = np.sum(self.px**2*self.W0)
        pW = self.Wx.dot(self.px)
        for i in range(1,walk+1):
            K[i] = K[i-1] + lamb**i * np.sum(pW*self.px)
            pW = self.Wx.dot(pW)
        if self.debug:
            print('CPU - K    : %f' %(time()-t0))
        return K

    ##############################################################
    #
    #  CPU Routines
    #
    ##############################################################

    def compute_kron_mat(self,g1,g2):
        """Kroenecker matrix calculation edges pssm similarity.

        Args:
            g1 (iScore.Graph): first graph
            g2 (iScore.Graph): second graph

        """
        t0 = time()
        row,col,weight = [],[],[]
        n1,n2 = g1.num_edges,g2.num_edges
        N = n1*n2

        # double the edges index for g1
        index1 = np.vstack((g1.edges_index,np.flip(g1.edges_index,axis=1)))
        index2 = g2.edges_index

        # double the pssm edges for g1
        pssm1 = np.vstack((g1.edges_pssm,np.hstack((g1.edges_pssm[:,20:],g1.edges_pssm[:,:20]))))
        pssm2 = g2.edges_pssm

        # compute the weight
        if self.method == 'iter':
            weight  = np.array([ self._rbf_kernel(p[0],p[1]) for p in itertools.product(*[pssm1,pssm2]) ])
            ind     = np.array([ self._get_index(k[0],k[1],g2.num_nodes)  for k in itertools.product(*[index1,index2])])

        elif self.method == 'combvec':
            weight = self._rbf_kernel_combvec(pssm1,pssm2)
            ind    = self._get_index_combvec(index1,index2,g2.num_nodes)

        elif self.method == 'vect':
            weight = self._rbf_kernel_vectorized(pssm1,pssm2)
            ind    = self._get_index_vect(index1,index2,g2.num_nodes)

        else:
            raise ValueError('Method %s not recognized' %self.method)

        # final size
        n_nodes_prod = g1.num_nodes*g2.num_nodes

        # instead of taking the transpose we duplicate
        # the weight and indexes (with switch)
        _manual_transpose_ = True
        if _manual_transpose_:

            weight = np.concatenate((weight,weight))
            ind = np.vstack((ind,np.flip(ind,axis=1)))
            index = ( ind[:,0],ind[:,1] )

            # create the matrix
            self.Wx = sp_sparse.coo_matrix( (weight,index),shape=( n_nodes_prod,n_nodes_prod ) )

        if not _manual_transpose_:
            index = ( ind[:,0],ind[:,1] )
            self.Wx = sp_sparse.coo_matrix( (weight,index),shape=( n_nodes_prod,n_nodes_prod ) )
            self.Wx += self.Wx.transpose()

        if self.debug:
            print('CPU - Kron : %f' %(time()-t0))

    def compute_px(self,g1,g2,cutoff=0.5):
        """Calculation of the Px vector from the nodes info.

        Args:
            g1 (iScore.Graph): first graph
            g2 (iScore.Graph): second graph
            cutoff (float, optional): if px[i]<cuoff -> px[i]=0
        """
        t0 = time()
        #n1,n2 = g1.num_nodes,g2.num_nodes
        self.px = [t[0]*t[1] if (float(t[0])>cutoff or float(t[1])>cutoff) else 0 for t in itertools.product(*[g1.nodes_info_data,g2.nodes_info_data])]
        
        if self.debug:
            print('CPU - Px   : %f' %(time()-t0))

    def compute_W0(self,g1,g2,method='vect'):
        """Calculation of t W0 vector from the nodes pssm similarity

        Args:
            g1 (iScore.Graph): first graph
            g2 (iScore.Graph): second graph
            method (str, optional): options: iter, combvec, vect (default vect)

        """
        t0 = time()

        if self.method == 'iter':
            self.W0  = np.array([ self._rbf_kernel(p[0],p[1]) for p in itertools.product(*[g1.nodes_pssm_data,g2.nodes_pssm_data]) ])

        elif self.method == 'combvec':
            self.W0  = self._rbf_kernel_combvec(g1.nodes_pssm_data,g2.nodes_pssm_data)

        elif self.method == 'vect':
            self.W0 = self._rbf_kernel_vectorized(g1.nodes_pssm_data,g2.nodes_pssm_data)

        else:
            raise ValueError('Method %s not recognized' %self.method)

        if self.debug:
            print('CPU - W0   : %f' %(time()-t0))

    @staticmethod
    def _combvec(a1,a2,axis=0):
        """Returns all the combination of the column vectors contained in a1 and a2.

        Args:
            a1 (np.array): matrix of vectors
            a2 (np.array): matrix of vectors
            axis (int, optional): axis for the combination

        Returns:
            np.array: matrix holding the all the combination of the vectors
        """
        n1,m1 = a1.shape
        n2,m2 = a2.shape
        if axis == 0:
            return np.vstack((np.repeat(a1,m2,axis=1),np.tile(a2,(1,m1))))
        if axis == 1:
            return np.hstack((np.repeat(a1,n2,axis=0),np.tile(a2,(n1,1))))

    @staticmethod
    def _rbf_kernel(data1,data2,sigma=10):
        """Kernel for the edges pssm similarity calculation.
        Used in the iter method.

        Args:
            data1 (TYPE): pssm data 1
            data2 (TYPE): pssm dta 2
            sigma (int, optional): exponent of the exponetial

        Returns:
            float: value of the rbk kernel
        """

        delta = np.sum((data1-data2)**2)
        beta = 2*sigma**2
        return np.exp(-delta/beta)

    def _rbf_kernel_combvec(self,data1,data2,sigma=10):
        """kernel for edge similarity computed with the combvec method

        Args:
            data1 (TYPE): pssm data 1
            data2 (TYPE): pssm dta 2
            sigma (int, optional): exponent of the exponetial

        Returns:
            np.array: value of the rbk kernel for all the pairs
        """
        k = data1.shape[1]
        data = self._combvec(data1,data2,axis=1)
        data = np.sum((data[:,:k]-data[:,k:])**2,1)
        beta = 2*sigma**2
        return np.exp(-data/beta)

    @staticmethod
    def _rbf_kernel_vectorized(data1,data2,sigma=10):
        """kernel for edge similarity computed with the vectorized method

        Args:
            data1 (TYPE): pssm data 1
            data2 (TYPE): pssm dta 2
            sigma (int, optional): exponent of the exponetial

        Returns:
            np.array: value of the rbk kernel for all the pairs
        """
        delta = -2*np.dot(data1,data2.T) + np.sum(data1**2,axis=1)[:,None] + np.sum(data2**2,axis=1)
        beta = 2*sigma**2
        return np.exp(-delta/beta).reshape(-1)

    @staticmethod
    def _get_index(index1,index2,size2):
        """Get the index in the bigraph iter method

        Args:
            index1 (list(int)): List of the edge index in the first graph
            index1 (list(int)): List of the edge index in the second graph
            size2 (int): Number of nodes in the second graph

        Returns:
            list(int): List of index in the bigraph
        """
        index = np.array(index1.tolist()) * size2 + np.array(index2.tolist())
        return index.tolist()

    def _get_index_combvec(self,index1,index2,size2):
        """Get the index in the bigraph combvec method

        Args:
            index1 (list(int)): List of the edge index in the first graph
            index1 (list(int)): List of the edge index in the second graph
            size2 (int): Number of nodes in the second graph

        Returns:
            list(int): List of index in the bigraph
        """
        index = self._combvec(index1,index2,axis=1)
        return index[:,:2]*float(size2) + index[:,2:]

    @staticmethod
    def _get_index_vect(index1,index2,size2):
        """Get the index in the bigraph vect method

        Args:
            index1 (list(int)): List of the edge index in the first graph
            index1 (list(int)): List of the edge index in the second graph
            size2 (int): Number of nodes in the second graph

        Returns:
            list(int): List of index in the bigraph
        """
        index1 = index1*float(size2)

        return np.hstack((
        (index1[:,0][:,np.newaxis] + index2[:,0]).reshape(-1,1),
        (index1[:,1][:,np.newaxis] + index2[:,1]).reshape(-1,1)))


    ##############################################################
    #
    #  CUBLAS Routines
    #
    ##############################################################


    def compute_kron_mat_cublas(self,g1,g2): # pragma: no cover
        """Kroenecker matrix calculation edges pssm similarity.

        Args:
            g1 (iScore.Graph): first graph
            g2 (iScore.Graph): second graph

        """
        t0 = time()
        row,col,weight = [],[],[]
        n1,n2 = g1.num_edges,g2.num_edges
        N = n1*n2

        # double the edges index for g1
        index1 = np.vstack((g1.edges_index,np.flip(g1.edges_index,axis=1)))
        index2 = g2.edges_index

        # double the pssm edges for g1
        pssm1 = np.vstack((g1.edges_pssm,np.hstack((g1.edges_pssm[:,20:],g1.edges_pssm[:,:20]))))
        pssm2 = g2.edges_pssm

        # compute the weight
        weight = self._rbf_kernel_vectorized_cublas(pssm1,pssm2)
        ind    = self._get_index_vect(index1,index2,g2.num_nodes)

        # final size
        n_nodes_prod = g1.num_nodes*g2.num_nodes

        # instead of taking the transpose we duplicate
        # the weight and indexes (with switch
        weight = np.concatenate((weight,weight))
        ind = np.vstack((ind,np.flip(ind,axis=1)))
        index = ( ind[:,0],ind[:,1] )

        # create the matrix
        self.Wx = sp_sparse.coo_matrix( (weight,index),shape=( n_nodes_prod,n_nodes_prod ) )

        if self.debug:
            print('GPU - Kron : %f' %(time()-t0))


    @staticmethod
    def _rbf_kernel_vectorized_cublas(data1,data2,sigma=10): # pragma: no cover
        """kernel for edge similarity computed with the vectorized method

        Args:
            data1 (TYPE): pssm data 1
            data2 (TYPE): pssm dta 2
            sigma (int, optional): exponent of the exponetial

        Returns:
            np.array: value of the rbk kernel for all the pairs
        """
        beta = 2*sigma**2
        d1_ = gpuarray.to_gpu(data1.astype(np.float32))
        d2_ = gpuarray.to_gpu(data2.astype(np.float32))
        mgpu  = -2*culinalg.dot(d1_,d2_,transa='N',transb='T')
        vgpu = cumisc.sum(d1_**2,axis=1)[:,None]
        cumisc.add_matvec(mgpu,vgpu,out=mgpu)

        vgpu = cumisc.sum(d2_**2,axis=1)
        cumisc.add_matvec(mgpu,vgpu,out=mgpu)

        mcpu = mgpu.get()
        return np.exp(-mcpu/beta).reshape(-1)

    ##############################################################
    #
    #  CUDA Routines
    #
    ##############################################################

    def compile_kernel(self): # pragma: no cover
        """Compile the file containing the CUDA kernels."""

        t0 = time()
        kernel_code = open(self.kernel, 'r').read()
        self.mod = compiler.SourceModule(kernel_code)
        if self.debug:
            print('GPU - Kern : %f' %(time()-t0))

    def compute_kron_mat_cuda(self,g1,g2,kernel_name='create_kron_mat',gpu_block=None): # pragma: no cover
        """kronecker matrix with the edges pssm

        Args:
            g1 (iScore.Graph): first graph
            g2 (iScore.Graph): second graph
            kernel_name (str): name of the kernel to use
            gpu_block (None, optional): Size of the GPU block
        """
        n1 = g1.num_edges
        n2 = g2.num_edges
        n_edges_prod = 2*n1*n2

        # get the gpu block size if specified
        if gpu_block is not None:
            block = gpu_block
        else:
            block = self.gpu_block
        dim = (n1,n2,1)
        grid = tuple([int(np.ceil(n/t)) for n,t in zip(dim,block)])

        # start timer
        t0 = time()
        driver.Context.synchronize()
        create_kron_mat_gpu = self.mod.get_function(kernel_name)

        # put the raw pssm on the GPU
        pssm1 = gpuarray.to_gpu(np.array(g1.edges_pssm).astype(np.float32))
        pssm2 = gpuarray.to_gpu(np.array(g2.edges_pssm).astype(np.float32))

        # we have to put the index on the gpu as well
        ind1 = gpuarray.to_gpu(np.array(g1.edges_index).astype(np.int32))
        ind2 = gpuarray.to_gpu(np.array(g2.edges_index).astype(np.int32))

        # create the gpu arrays only if we have to
        # i.e. in case we run the calculation once (test or tune)
        # in other cases the weigh and index are booked in self.run()
        if not hasattr(self,'weight_product'):
            self.weight_product = gpuarray.zeros(n_edges_prod, np.float32)
            self.index_product = gpuarray.zeros((n_edges_prod,2), np.int32)

        driver.Context.synchronize()
        if self.debug:
            print('GPU - Mem  : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))

        # use the combvec kernel
        t0 = time()
        create_kron_mat_gpu (ind1,ind2,
                             pssm1,pssm2,
                             self.index_product,self.weight_product,
                             n1,n2,g2.num_nodes,
                             block=block,grid=grid)

        # extract the data
        # restrict to the ones calculated here
        ind = self.index_product.get()
        w = self.weight_product.get()[:n_edges_prod]

        # final size
        n_nodes_prod = g1.num_nodes*g2.num_nodes

        # create the matrix
        tt = time()

        # replaced the transpose with
        # doubling of weight and index (with switch)
        w = np.concatenate((w,w))
        ind = np.vstack((ind,np.flip(ind,axis=1)))
        index = ( ind[:,0],ind[:,1])
        self.Wx = sp_sparse.coo_matrix( (w,index),shape=( n_nodes_prod,n_nodes_prod ) )


        #driver.Context.synchronize()
        if self.debug:
            print('GPU - Kron : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))

    def compute_px_cuda(self,g1,g2,gpu_block=None): # pragma: no cover
        """Calculation of the PX vector.

        Args:
            g1 (iScore.Graph): first graph
            g2 (iScore.Graph): second graph
            gpu_block (None, optional): Size of the GPU block
        """
        t0 = time()
        driver.Context.synchronize()

        create_p_vect = self.mod.get_function('create_p_vect')
        info1 = gpuarray.to_gpu(np.array(g1.nodes_info_data).astype(np.float32))
        info2 = gpuarray.to_gpu(np.array(g2.nodes_info_data).astype(np.float32))

        n_nodes_prod = g1.num_nodes*g2.num_nodes
        pvect = gpuarray.zeros(n_nodes_prod,np.float32)

        if gpu_block is not None:
            block = gpu_block
        else:
            block = self.gpu_block
        dim = (g1.num_nodes,g2.num_nodes,1)
        grid = tuple([int(np.ceil(n/t)) for n,t in zip(dim,block)])

        create_p_vect(info1,info2,pvect,g1.num_nodes,g2.num_nodes,block=block,grid=grid)
        self.px = pvect.get()
        driver.Context.synchronize()
        if self.debug:
            print('GPU - Px   : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))

    def compute_W0_cuda(self,g1,g2,gpu_block=None): # pragma: no cover
        """Calculation of the W0 matrix from the nodes pssm.

        Args:
            g1 (iScore.Graph): first graph
            g2 (iScore.Graph): second graph
            gpu_block (None, optional): Size of the GPU block
        """
        t0 = time()
        driver.Context.synchronize()

        compute = self.mod.get_function('create_nodesim_mat')
        pssm1 = gpuarray.to_gpu(np.array(g1.nodes_pssm_data).astype(np.float32))
        pssm2 = gpuarray.to_gpu(np.array(g2.nodes_pssm_data).astype(np.float32))
        n_nodes_prod = g1.num_nodes*g2.num_nodes
        w0 = gpuarray.zeros(n_nodes_prod,np.float32)

        if gpu_block is not None:
            block = gpu_block
        else:
            block = self.gpu_block
        dim = (g1.num_nodes,g2.num_nodes,1)
        grid = tuple([int(np.ceil(n/t)) for n,t in zip(dim,block)])

        compute(pssm1,pssm2,w0,g1.num_nodes,g2.num_nodes,block=block,grid=grid)
        self.W0 = w0.get()
        driver.Context.synchronize()
        if self.debug:
            print('GPU - W0   : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))

    def tune_kernel(self,func='create_kron_mat',test_all_func=False): # pragma: no cover
        """Method to tune the kernel using the KernelTuner

        Args:
            func (str, optional): Name of the function to tune
            test_all_func (bool, optional): test all the functions (default False)
        """
        try:
            from kernel_tuner import tune_kernel
        except ModuleNotFoundError :
            print('Install the Kernel Tuner : \n \t\t pip install kernel_tuner')
            print('http://benvanwerkhoven.github.io/kernel_tuner/')


        test_name = list(self.test_graphs.keys())[0]
        train_name = list(self.train_graphs.keys())[0]

        g1 = self.test_graphs[test_name]
        g2 = self.train_graphs[train_name]

        tune_params = OrderedDict()
        tune_params['block_size_x'] = [2,4,8,16,32,64,128]
        tune_params['block_size_y'] = [2,4,8,16,32,64,128]

        kernel_code = open(self.kernel, 'r').read()
        tunable_kernel = self._tunable_kernel(kernel_code)

        try:

            if func == 'create_kron_mat' or test_all_func:

                func = 'create_kron_mat'
                print('\n')
                print('Tuning function %s from %s' %(func,self.kernel))
                print('-'*40)

                n1 = g1.num_edges
                n2 = g2.num_edges
                n_edges_prod = 2*n1*n2

                pssm1 = np.array(g1.edges_pssm).astype(np.float32)
                pssm2 = np.array(g2.edges_pssm).astype(np.float32)

                ind1 = np.array(g1.edges_index).astype(np.int32)
                ind2 = np.array(g2.edges_index).astype(np.int32)

                weight_product = np.zeros(n_edges_prod, np.float32)
                index_product = np.zeros((n_edges_prod,2), np.int32)

                dim = (n1,n2,1)
                args = [ind1,ind2,pssm1,pssm2,index_product,weight_product,n1,n2,g2.num_nodes]

                _ = tune_kernel(func,tunable_kernel,dim,args,tune_params)

            if func == 'create_nodesim_mat' or test_all_func:

                func = 'create_nodesim_mat'
                print('\n')
                print('Tuning function %s from %s' %(func,self.kernel))
                print('-'*40)

                pssm1 = np.array(g1.nodes_pssm_data).astype(np.float32)
                pssm2 = np.array(g2.nodes_pssm_data).astype(np.float32)
                n_nodes_prod = g1.num_nodes*g2.num_nodes
                w0 = np.zeros(n_nodes_prod,np.float32)

                dim = (g1.num_nodes,g2.num_nodes,1)
                args = [pssm1,pssm2,w0,g1.num_nodes,g2.num_nodes]

                _ = tune_kernel(func,tunable_kernel,dim,args,tune_params)

            if func == 'create_p_vect' or test_all_func:

                func = 'create_p_vect'
                print('\n')
                print('Tuning function %s from %s' %(func,self.kernel))
                print('-'*40)

                info1 = np.array(g1.nodes_info_data).astype(np.float32)
                info2 = np.array(g2.nodes_info_data).astype(np.float32)
                n_nodes_prod = g1.num_nodes*g2.num_nodes
                pvect = np.zeros(n_nodes_prod,np.float32)

                dim = (g1.num_nodes,g2.num_nodes,1)
                args = [info1,info2,pvect,g1.num_nodes,g2.num_nodes]

                _ = tune_kernel(func,tunable_kernel,dim,args,tune_params)

        except:
            print('Function %s not found in %s' %(func,self.kernel))

    @staticmethod
    def _tunable_kernel(kernel): # pragma: no cover
        """Transform the kernel to a tunable one.

        Args:
            kernel (str): Kernel

        Returns:
            str: ne kernel
        """
        switch_name = { 'blockDim.x' : 'block_size_x', 'blockDim.y' : 'block_size_y' }
        for old,new in switch_name.items():
            kernel = kernel.replace(old,new)
        return kernel


def iscore_kernel(testID=None,trainID=None,
                  test_graph='./graph', train_graph='./graph',\
                  train_archive=None,
                  check=None, outfile='kernel.pkl',test=False,
                  lamb=1, walk=4, method='vect',
                  tune_kernel=False,func='all',cuda=False, gpu_block=[8,8,1]):

    # init and load the data
    ker = Kernel(testIDs=testID,test_graph = test_graph,
                 trainIDs=trainID,train_graph=train_graph,
                 train_archive=train_archive,
                 gpu_block=tuple(gpu_block),method=method)
    ker.import_from_mat()

    # get the path of the check file
    checkfile = ker.get_check_file(check)

    # only tune the kernel
    if tune_kernel:
        ker.tune_kernel(func=func,test_all_func=func)

    # run the entire calculation
    else :
        ker.run(lamb=lamb,
               walk=walk,
               outfile=outfile,
               cuda=cuda,
               gpu_block=tuple(gpu_block),
               check=checkfile,
               test=test)

def iscore_kernel_mpi(testID=None,trainID=None,
                      test_graph='./graph', train_graph='./graph',
                      train_archive=None,
                      check=None, outfile='kernel.pkl',test=False,
                      lamb=1, walk=4, method='vect',rank=0,size=1):

    # init and load the data
    ker = Kernel(testIDs=testID,test_graph = test_graph,
                 trainIDs=trainID,train_graph=train_graph,
                 train_archive=train_archive,method=method)

    ker.import_from_mat(mpi_rank=rank,mpi_size=size)

    # get the path of the check file
    checkfile = ker.get_check_file(check)

    # run the entire calculation
    ker.run(lamb=lamb,
           walk=walk,
           outfile=outfile,
           check=checkfile,
           test=test,
           mpi_rank=rank,
           mpi_size=size)
