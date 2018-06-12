import os
import numpy as np
import scipy.io as spio
import pickle


class Graph(object):

    def __init__(self,fname=None,file_type=None,
                      name=None,nodes_pssm=None,nodes_info=None,edges_index=None):

        if fname is not None:
            self.load(fname,file_type)
        else:
            self.name = name
            self.nodes_pssm_data = nodes_pssm
            self.nodes_info_data = nodes_info
            self.edges_index = edges_index

            if self._valid_data():
                self._process_data()

        #self.print()

    def load(self,fname,file_type=None):

        if not os.path.isfile(fname):
            raise FileNotFoundError('File %s not found' %fname)

        # determine file type if it's not given
        if file_type is None:
            dict_ext = {'matlab':['.mat','.MAT'],
                        'pickle':['.pkl','.pckl'] }
            ext = os.path.splitext(fname)[1]
            for k,v in dict_ext.items():
                if ext in v:
                    file_type = k
                    break

        # import the graph
        if file_type == 'matlab':
            self._load_from_matlab(fname)
        elif file_type == 'pickle':
            self._load_from_pickle(fname)

    def _load_from_matlab(self,fname):

        data = spio.loadmat(fname,squeeze_me=True)['G']

        self.name = ext = os.path.splitext(fname)[0]
        self.nodes_pssm_data = np.array([p.tolist() for p in data['Nodes'][()]['pssm'][()]])
        self.nodes_info_data = data['Nodes'][()]['info'][()]
        self.edges_index = np.array(data['Edges'][()]['idx'][()])-1
        self._process_data()


    def _load_from_pickle(self,fname):

        data = pickle.load(open(fname,'rb'))

        self.name = ext = os.path.splitext(fname)[0]
        self.nodes_pssm_data = data.nodes_pssm_data
        self.nodes_info_data = data.nodes_info_data
        self.edges_index = np.array(data.edges_index)
        self._process_data()

    def _process_data(self):

        self.num_nodes = np.int32(len(self.nodes_info_data))
        self.num_edges = np.int32(len(self.edges_index))

        self.edges_pssm = []
        for ind in self.edges_index:
            self.edges_pssm.append( self.nodes_pssm_data[ind[0]].tolist() + self.nodes_pssm_data[ind[1]].tolist()  )
        self.edges_pssm = np.array(self.edges_pssm)


    def _valid_data(self):
        data = [self.nodes_pssm_data,self.nodes_info_data,self.edges_index]
        return np.all(data != None)

    def pickle(self,fname):
        pickle.dump(self,open(fname,'wb'))

    def print(self):
        print('='*40)
        print('=   ', self.name)
        print('=    %d nodes' %self.num_nodes)
        print('=    %d edges' %self.num_edges)
        print('='*40)
        print('\n ----- Nodes')
        for iNode in range(self.num_nodes):
            for iP in range(20):
                print('% 2d' %self.nodes_pssm_data[iNode][iP],end=' ')
            print(self.nodes_info_data[iNode])
        print('\n ----- Edges')
        for iE in range(self.num_edges):
            print('%02d <--> %d' %(self.edges_index[iE][0],self.edges_index[iE][1]))
        print('='*40)
        print('\n')

    def compare(self, gcheck,verbose = True):

        same = 1

        if self.num_nodes != gcheck.num_nodes:
            if verbose:
                print('Graphs %s and %s have different number of nodes' %(self.name,gcheck.name))
            same = 0
        if self.num_edges != gcheck.num_edges:
            if verbose:
                print('Graphs %s and %s have different number of edges' %(self.name,gcheck.name))
            same = 0

        if same:

            pssm = tuple(self.nodes_pssm_data.tolist())
            pssm_check = tuple(gcheck.nodes_pssm_data.tolist())

            nodes_mapping = {}
            skip = 0
            for i in range(self.num_nodes):
                try:
                    ind = [k for k,x in enumerate(pssm_check) if x == pssm[i]]
                    #ind = pssm_check.index(pssm[i])
                    if len(ind) == 1:
                        ind = ind[0]
                        nodes_mapping[ind] = i
                    else:
                        print('Warning multiple nodes with idential PSSM')
                        skip = 1
                except:
                    if verbose:
                        print('Node not found')
                    same = 0
            if not skip:
                edges = tuple(e.tolist() for e in self.edges_index)

                try:
                    for i in range(gcheck.num_edges):
                        x,y = gcheck.edges_index[i]
                        a,b = nodes_mapping[x],nodes_mapping[y]
                        if [a,b] not in edges and [b,a] not in edges:
                            if verbose:
                                print('Edges %d %d not found in original graph' %(a,b))
                                print('corresponds to edges %d %d in reference graph' %(x,y))
                            same = 0
                except:
                    if verbose:
                        print('Node Mapping Issues')
                    same = 0
            if same and verbose:
                print('Graphs %s and %s are identical' %(self.name,gcheck.name))

        return same

    def reconstruct_residue_graphs(self,pssm_dict):

        pssm = {}
        for chain in ['A','B']:
            pssm[chain] = self.read_PSSM_data(pssm_dict[chain])


         # residue name translation dict
        resmap = {
        'A' : 'ALA', 'R' : 'ARG', 'N' : 'ASN', 'D' : 'ASP', 'C' : 'CYS', 'E' : 'GLU', 'Q' : 'GLN',
        'G' : 'GLY', 'H' : 'HIS', 'I' : 'ILE', 'L' : 'LEU', 'K' : 'LYS', 'M' : 'MET', 'F' : 'PHE',
        'P' : 'PRO', 'S' : 'SER', 'T' : 'THR', 'W' : 'TRP', 'Y' : 'TYR', 'V' : 'VAL',
        'B' : 'ASX', 'U' : 'SEC', 'Z' : 'GLX'
        }
        resmap_inv = {v: k for k, v in resmap.items()}


        self.pssm_name_dict = {}

        for chain in ['A','B']:
            for iL in range(len(pssm[chain])):
                if pssm[chain][iL][1] in resmap:
                    res = '_'.join([chain,pssm[chain][iL][0],resmap[pssm[chain][iL][1]]])
                else:
                    print('What is that ',pssm[chain][iL][1])
                    exit()
                tmp = (tuple([float(v) for v in pssm[chain][iL][4:]]))
                if tmp not in self.pssm_name_dict:
                    self.pssm_name_dict[tmp] = res
                else:
                    print('Identical PSSM for different residue',tmp)
                    if not isinstance(self.pssm_name_dict[tmp],list):
                        self.pssm_name_dict[tmp]  = [self.pssm_name_dict[tmp]]
                    self.pssm_name_dict[tmp].append(res)

    def read_PSSM_data(self,fname):
        """Read the PSSM data."""

        f = open(fname,'r')
        data = f.readlines()
        f.close()

        filters = (lambda x: len(x.split())>0, lambda x: x.split()[0].isdigit())
        return list(map(lambda x: x.split(),list(filter(lambda x: all(f(x) for f in filters), data))))