import os
import numpy as np
import scipy.io as spio
import pickle
import h5py
from pdb2sql import pdb2sql
from pdb2sql import interface
from Bio import pairwise2
import warnings

class Graph(object):

    def __init__(self,fname=None,file_type=None, chain_label=['A','B'],
                      name=None,nodes_pssm=None,
                      nodes_info=None,edges_index=None):
        """Graph object corresponding to a given PDB file.

        Example:

        >>> from iScore.graph import Graph
        >>> # load an exiting graph
        >>> g = Graph('graph_file.pkl')
        >>> # create a new one
        >>> graph = Graph(name = name,nodes_pssm = nodes_pssm,nodes_info = nodes_info, edges_index = edges)

        Args:
            fname (None, optional): File name of an existing graph
            file_type (None, optional): File type of fname ('.mat','.MAT','.pkl','.pckl')
            name (None, optional): Name of the conformation
            nodes_pssm (None, optional): Node PSSM data
            nodes_info (None, optional): Node info data
            edges_index (None, optional): index of the edges in the graph
        """

        self.fname = fname

        if fname is not None:
            self.load(fname,file_type)
        else:
            self.name = name
            self.chain_label = chain_label
            self.nodes_pssm_data = nodes_pssm
            self.nodes_info_data = nodes_info
            self.edges_index = edges_index

            if self._valid_data():
                self._process_data()

    def load(self,fname,file_type=None):
        """Load an existing graph file.

        Args:
            fname (str): name of the file
            file_type (None or str, optional): file extension
                                               If None will try to find a suitable file

        Raises:
            FileNotFoundError: If the file is not found
        """
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
        """Load a matlab graph file

        Args:
            fname (str): name of the file
        """
        data = spio.loadmat(fname,squeeze_me=True)['G']

        self.name = os.path.splitext(fname)[0]
        self.nodes_pssm_data = np.array([p.tolist() for p in data['Nodes'][()]['pssm'][()]])
        self.nodes_info_data = data['Nodes'][()]['info'][()]
        self.edges_index = np.array(data['Edges'][()]['idx'][()])-1
        self._process_data()


    def _load_from_pickle(self,fname):
        """Load a pickle file.

        Args:
            fname (str): File name
        """
        f =open(fname,'rb')
        data = pickle.load(f)
        f.close()

        self.name = ext = os.path.splitext(fname)[0]
        self.nodes_pssm_data = data.nodes_pssm_data
        self.nodes_info_data = data.nodes_info_data
        self.edges_index = np.array(data.edges_index)
        self._process_data()

    def _process_data(self):
        """Creates all the data needed for the graph."""

        self.num_nodes = np.int32(len(self.nodes_info_data))
        self.num_edges = np.int32(len(self.edges_index))

        self.edges_pssm = []
        for ind in self.edges_index:
            self.edges_pssm.append( self.nodes_pssm_data[ind[0]].tolist() + self.nodes_pssm_data[ind[1]].tolist()  )
        self.edges_pssm = np.array(self.edges_pssm)


    def _valid_data(self):
        """Check that the data is correct and does not contain Nones

        Returns:
            bool: boolean to ensure the validity of the data
        """
        data = [self.nodes_pssm_data,self.nodes_info_data,self.edges_index]
        return np.all(data != None)

    def pickle(self,fname):
        """Create a pickle file containing the graph

        Args:
            fname (str): filename
        """
        f = open(fname,'wb')
        pickle.dump(self,f)
        f.close()

    def print(self): #pragma: no cover
        """Print the graph to screen for debugging."""

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
        """Compare two graphs to see if they are identical.
        This can be very usefull to check if the graph outputed
        here are identical to the ones obtained with the matlab code

        Args:
            gcheck (Graph): Graph instance to be compared with the current graph
            verbose (bool, optional): Print all sorts of information

        Returns:
            Bool: 1 if graphs are identical 0 otherwise
        """
        same = 1

        if self.num_nodes != gcheck.num_nodes:
            if verbose: #pragma: no cover
                print('Graphs %s and %s have different number of nodes' %(self.name,gcheck.name))
            same = 0
        if self.num_edges != gcheck.num_edges:
            if verbose: #pragma: no cover
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
                except Exception as e:
                    if verbose:
                        print(e)

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
                print('Graphs %s and %s are identical' %(self.fname,gcheck.fname))

        return same

    def reconstruct_residue_graphs(self,pssm_dict):
        """Build the graph with the residue name/number corresponding to the PDB

        Args:
            pssm_dict (dict): pssm information
        """
        pssm = {}
        for chain in self.chain_label:
            pssm[chain] = self.read_PSSM_data(pssm_dict[chain])


         # residue name translation dict
        resmap = {
        'A' : 'ALA', 'R' : 'ARG', 'N' : 'ASN', 'D' : 'ASP', 'C' : 'CYS', 'E' : 'GLU', 'Q' : 'GLN',
        'G' : 'GLY', 'H' : 'HIS', 'I' : 'ILE', 'L' : 'LEU', 'K' : 'LYS', 'M' : 'MET', 'F' : 'PHE',
        'P' : 'PRO', 'S' : 'SER', 'T' : 'THR', 'W' : 'TRP', 'Y' : 'TYR', 'V' : 'VAL',
        'B' : 'ASX', 'U' : 'SEC', 'Z' : 'GLX'
        }

        self.pssm_name_dict = {}

        for chain in self.chain_label:
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

    @staticmethod
    def read_PSSM_data(fname):
        """Read the PSSM data."""

        f = open(fname,'r')
        data = f.readlines()
        f.close()

        filters = (lambda x: len(x.split())>0, lambda x: x.split()[0].isdigit())
        return list(map(lambda x: x.split(),list(filter(lambda x: all(f(x) for f in filters), data))))



class GenGraph():

    def __init__(self,pdbfile,pssmfile, aligned=True, export=True,
                 outname=None, cutoff=6.0, h5file = None):
        """Generates a graph from pdb and pssm files.

        Example:

        >>> pdb = './1ATN.pdb'
        >>> pssm = {'A':'./1ATN.A.pdb.pssm','B':'./1ATN.B.pdb.pssm'}
        >>> outfile = './1ATN.pkl'
        >>> check_file = './1ATN.mat'
        >>> g = GenGraph(pdb,pssm,export=True,outname=outfile)
        >>> gcheck = Graph(check_file)
        >>> check = g.compare(gcheck)

        Args:
            pdbfile (str):  pdb files path
            pssmfile (str): pssm files path
            aligned (bool, optional): PSSM aligned to PDB
            export (bool, optional): Export graph files
            outname (None, optional): Directory containing all the exported graphs
            cutoff (float, optional): Cutoff distance to select contact atoms
            h5file (file handle, optional): hdf5 file handle to store the graph (None if no export)
        """

        # pdb file
        self.pdbfile = pdbfile
        self.pdb = pdb2sql(self.pdbfile)

        # residue name translation dict
        self.resmap = {
        'A' : 'ALA', 'R' : 'ARG', 'N' : 'ASN', 'D' : 'ASP', 'C' : 'CYS', 'E' : 'GLU', 'Q' : 'GLN',
        'G' : 'GLY', 'H' : 'HIS', 'I' : 'ILE', 'L' : 'LEU', 'K' : 'LYS', 'M' : 'MET', 'F' : 'PHE',
        'P' : 'PRO', 'S' : 'SER', 'T' : 'THR', 'W' : 'TRP', 'Y' : 'TYR', 'V' : 'VAL',
        'B' : 'ASX', 'U' : 'SEC', 'Z' : 'GLX'
        }
        self.resmap_inv = {v: k for k, v in self.resmap.items()}

        # pssm data
        self.pssm = {}
        self.chain_label = list(pssmfile.keys())
        for chain in self.chain_label:
            self.pssm[chain] = self.read_PSSM_data(pssmfile[chain])

        # check format
        if not self.check_pssm_format(aligned):
            print('--> PSSM format issue.\n--> Check if they are aligned or not.\n--> If they are use: iScore.generate --aligned')
            exit()

        # cutoff for the contact
        self.cutoff = cutoff

        # out file
        if outname is None:
            outname = os.path.splitext(pdbfile)[0] + '.pckl'

        # if the pssm data where not aligned
        if not aligned:
            self.get_aligned_pssm()
        else:
            self.process_aligned_pssm()

        self.construct_graph()

        if export:
            self.export_graph(outname)

        if h5file is not None:
            self.toh5(h5file)

    def check_pssm_format(self,aligned):
        """Check the format of the PSSM files

        Args:
            aligned (bool): Are the PSSM files supposed to be aligned to the PDBs

        Returns:
            bool: 1 if format is ok 0 otherwise
        """

        key = list(self.pssm.keys())[0]
        if aligned:
            return len(self.pssm[key][0]) == 25
        else:
            return len(self.pssm[key][0]) == 44 #48 ?


    def read_PSSM_data(self,fname):
        """Read the PSSM data."""

        f = open(fname,'r')
        data = f.readlines()
        f.close()

        filters = (lambda x: len(x.split())>0, lambda x: x.split()[0].isdigit())
        return list(map(lambda x: x.split(),list(filter(lambda x: all(f(x) for f in filters), data))))

    def get_aligned_pssm(self):
        """Align the PSSM file to the pdb.
        Untested in a while ...
        """

        self._align_sequences()

        self.aligned_pssm = {}
        self.aligned_ic = {}
        for chain in self.chain_label:

            iResPDB,iResPSSM = 0,0
            pdbres = [(numb,name) for numb,name in self.pdb.get('resSeq,resName',chainID=chain)]
            pdbres = [v for v in dict(pdbres).items()]

            for resPDB,resPSSM in zip(self.seq_aligned['pdb'][chain], self.seq_aligned['pssm'][chain]):

                if resPSSM == '-' and resPDB != '-':
                    self.aligned_pssm[(chain,)+pdbres[iResPDB]] = None
                    self.aligned_ic[(chain,)+pdbres[iResPDB]] = None
                    iResPDB += 1

                if resPSSM != '-' and resPDB == '-':
                    iResPSSM += 1

                if resPSSM != '-' and resPDB != '-':
                    self.aligned_pssm[(chain,)+pdbres[iResPDB]] = self.pssm[chain][iResPSSM][2:23]
                    self.aligned_ic[(chain,)+pdbres[iResPDB]] = self.pssm[chain][iResPSSM][43]
                    iResPDB += 1
                    iResPSSM += 1

    def _align_sequences(self):
        """Aling the squence given in a PDN and its PSSM files."""

        self.seq_aligned = {'pdb':{},'pssm':{}}
        for chain in self.chain_label:
            pdb_seq = self._get_sequence(chain=chain)
            pssm_seq = ''.join( [data[1] for data in self.pssm[chain] ] )
            self.seq_aligned['pdb'][chain], self.seq_aligned['pssm'][chain] = self._get_aligned_seq(pdb_seq,pssm_seq)


    def _get_sequence(self,chain='A'):
        """Get the sequence of a given chain in a PDB

        Args:
            chain (str, optional): Chain ID ('A','B')

        Returns:
            str: 1 letter encoding sequence
        """
        data = []
        for numb,name in self.pdb.get('resSeq,resName',chainID=chain):
            if name in self.resmap_inv.keys():
                data.append((numb,self.resmap_inv[name]))
        return ''.join([v[1] for v in dict(data).items()])

    @staticmethod
    def _get_aligned_seq(seq1, seq2):
        """Align two sequnces using global alignment and return aligned sequences.
            Paramters of global alignment:
                match: 1
                mismtach: 0
                gap open: -2
                gap extend: -1

        Arguments:
            seq1 {str} -- 1st sequence.
            seq2 {str} -- 2nd sequence.

        Returns:
            [numpy array] -- seq1_ali, aligned sequence for seq1
            [numpy array] -- seq2_ali, aligned sequence for seq1
        """

        ali = pairwise2.align.globalxs(seq1, seq2, -2, -1)
        seq1_ali = np.array([i for i in ali[0][0]])
        seq2_ali = np.array([i for i in ali[0][1]])

        return seq1_ali, seq2_ali

    def process_aligned_pssm(self):
        """get the information from an aligned PSSM file."""

        self.aligned_pssm, self.aligned_ic = {},{}
        for chain in self.chain_label:
            for l in self.pssm[chain]:
                resi = int(l[0])
                resn = self.resmap[l[1]]
                self.aligned_pssm[(chain,resi,resn)] = l[4:24]
                self.aligned_ic[(chain,resi,resn)] = l[24]

    def construct_graph(self,verbose=False,print_res_pairs=False):
        """Construct the graph corresponding to a given PDB

        Args:
            verbose (bool, optional): print for debug
            print_res_pairs (bool, optional): print the residue contact pairs for debug
        """

        db = interface(self.pdbfile)
        res_contact_pairs = db.get_contact_residues(cutoff = self.cutoff,
                                                    allchains=True,
                                                    # chain1=self.chain_label[0],
                                                    # chain2=self.chain_label[1],
                                                    return_contact_pairs=True)


        # tag the non residues
        keys_to_pop = []
        for res in res_contact_pairs.keys():
            if res[2] not in self.resmap_inv:
                keys_to_pop.append(res)
                #res_contact_pairs.pop(res,None)
                print(res)
                warnings.warn('--> Residue not valid')

        # tag the ones that are not in PSSM
        for res in list(res_contact_pairs.keys()):
            if res not in self.aligned_pssm:
                keys_to_pop.append(res)
                #res_contact_pairs.pop(res,None)
                print(res)
                warnings.warn('--> Residue not found in PSSM file')

        # Remove the residue
        for res in keys_to_pop:
            if res in res_contact_pairs:
                res_contact_pairs.pop(res,None)

        # get a list of residues of chain B
        # automatically remove the ones that are not proper res
        # and the ones that are not in the PSSM
        nodesB = []
        for k,reslist in list(res_contact_pairs.items()):
            tmp_reslist = []
            for res in reslist:
                if res[2] in self.resmap_inv and res in self.aligned_pssm:
                    nodesB += [res]
                    tmp_reslist += [res]
                else:
                    print(res)
                    warnings.warn('--> Residue not found in PSSM file or Residue not recognized')
            res_contact_pairs[k] = tmp_reslist

        nodesB = sorted(set(nodesB))

        # make a list of nodes
        self.nodes = list(res_contact_pairs.keys()) + nodesB

        # get the edge numbering
        self.edges = []
        for key,val in res_contact_pairs.items():
            ind1 = self.nodes.index(key)
            for v in val:
                ind2 = self.nodes.index(v)
                self.edges.append([ind1,ind2])

        if print_res_pairs:
            for k,vs in res_contact_pairs.items():
                print(k)
                for v in vs:
                    print('\t\t',v)


    def get_graph(self,name=None):
        """Get the graph"""
        nodes_data,nodes_info = [],[]
        for res in self.nodes:

            pssm = self.aligned_pssm[res]
            nodes_data.append(pssm)

            ic = self.aligned_ic[res]
            nodes_info.append(ic)

        nodes_data = np.array(nodes_data)
        nodes_info = np.array(nodes_info)

        nodes_pssm = np.array(nodes_data).astype('int')
        nodes_info = np.array(nodes_info).astype('float')

        edges = np.array(self.edges)

        if name is None:
            name = os.path.splitext(self.pdbfile)[0]

        return Graph(name = name,
                      nodes_pssm = nodes_pssm,
                      nodes_info = nodes_info,
                      edges_index = edges)


    def export_graph(self,fname):
        """Export the graph of a given PDB

        Args:
            fname (str): file name
        """
        name = os.path.splitext(fname)[0]
        graph = self.get_graph(name)

        if graph.num_nodes > 0:
            graph.pickle(fname)
        else:
            print('Warning : Graph %s not exported (num_nodes = 0)' %fname)


    def toh5(self,f5):
        """Export the graph into an exisiting and open HDF5 file.

        Args:
            f5 (file handle): handle o the hdf5 file
        """

        grp_name = os.path.splitext(os.path.basename(self.pdbfile))[0]
        grp = f5.create_group(grp_name)
        grp.attrs['pdbfile'] = os.path.abspath(self.pdbfile)
        grp.create_dataset('nodes',data = np.array(self.nodes).astype('S'))
        grp.create_dataset('edges',data = np.array(self.edges))


def iscore_graph(pdb_path='./pdb/',pssm_path='./pssm/',select=None,
                 outdir='./graph/',aligned=True, export_hdf5=False):
    """Function called in the binary iScore.graph

    Args:
        pdb_path (str, optional): directory containing the pdb
        pssm_path (str, optional): directory containing the pssm
        select (None, optional): file containign the ID of the desired pdb. If None all PDBs are processed
        outdir (str, optional): Directory where to store the data
        aligned (bool, optional): Are the PSSM aligned
        export_hdf5(bool, optional): export the graph information to a single hdf5 file

    Raises:
        FileNotFoundError: If select has been specified but does not correspond to an existing file
        NotADirectoryError: If pdb_path or pssm_path were not found
    """

    # make sure that the dir containing the PDBs exists
    if not os.path.isdir(pdb_path):
        raise NotADirectoryError(pdb_path + ' is not a directory')

    # make sure that the dir containing the PSSMs exists
    if not os.path.isdir(pssm_path):
        raise NotADirectoryError(pssm_path + ' is not a directory')

    # create the outdir if necessary
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # check if we want to select a subset of PDBs
    if select is not None:
        if not os.path.isfile(select):
            raise FileNotFoundError(select + ' is not a file')
        else:
            with open(select,'r') as f:
                select = f.readlines()
    else:
        select = None

    # get the list of PDB names
    pdbs = list(filter(lambda x: x.endswith('.pdb'),os.listdir(pdb_path)))
    if select is not None:
        _tmp = []
        for s in select:
            s = s.strip('\n')
            for p in pdbs:
                if p.split('.')[0] == s:
                    _tmp.append(p)
        pdbs = _tmp

    # create the output file
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # get all the pssm files
    all_pssm_files = os.listdir(pssm_path)

    # loop over all the PDBs
    for name in pdbs:

        print('Creating graph of PDB %s' %name)

        # pdb name
        pdbfile = os.path.join(pdb_path,name)

        # mol name and base name
        mol_name = os.path.splitext(name)[0]

        # get the pssm files


        mol_pssm = list(filter(lambda x : x.startswith(mol_name+".") and x.endswith('.pdb.pssm'),all_pssm_files))
        if len(mol_pssm) == 0:
            print('--> Assuming global naming scheme for pssm files')
            mol_pssm = list(filter(lambda x : x.startswith(mol_name.split('_')[0]) and x.endswith('.pdb.pssm'),all_pssm_files))
        mol_pssm.sort()
        chain_label = [m.split('.')[-3] for m in mol_pssm]

        # print the pssm and chains found
        print(' --> Found the following chains and PSSM')
        pssm = dict()
        for c,f in zip(chain_label,mol_pssm):
            print('     %s : %s' %(c,f))
            pssm[c] = os.path.join(pssm_path,f)

        # append the pssm path to the files
        mol_pssm = [os.path.join(pssm_path,f) for f in mol_pssm]

        # check if the pssms exists
        #if os.path.isfile(mol_pssm[0]) and os.path.isfile(mol_pssm[1]):
        #    pssm = {chain_label[0]:mol_pssm[0],chain_label[1]:mol_pssm[1]}
        #else:
        #    raise FileNotFoundError(mol_pssm[0] + ' or ' + mol_pssm[1] + ' not found')

        # output file
        graphfile = os.path.join(outdir+mol_name+'.pckl')

        if export_hdf5:
            h5name = './graphs.hdf5'
            f5 = h5py.File(h5name,'a')
        else:
            f5 = None

        # create the graphs
        GenGraph(pdbfile,pssm,aligned=aligned,outname=graphfile,export=True,h5file=f5)

        if export_hdf5:
            f5.close()



def iscore_graph_mpi(pdb_path='./pdb/',pssm_path='./pssm/',select=None,
                     outdir='./graph/',aligned=True, export_hdf5=False,
                     rank=0,size=1,mpi_comm=None):
    """Function called in the binary iScore.graph.mpi

    Args:
        pdb_path (str, optional): directory containing the pdb
        pssm_path (str, optional): directory containing the pssm
        select (None, optional): file containign the ID of the desired pdb. If None all PDBs are processed
        outdir (str, optional): Directory where to store the data
        aligned (bool, optional): Are the PSSM aligned
        export_hdf5(bool, optional): export the graph information to a single hdf5 file

    Raises:
        FileNotFoundError: If select has been specified but does not correspond to an existing file
        NotADirectoryError: If pdb_path or pssm_path were not found
    """


    if rank == 0:

        # make sure that the dir containing the PDBs exists
        if not os.path.isdir(pdb_path):
            raise NotADirectoryError(pdb_path + ' is not a directory')
        # else:
        #     pdb_files = os.listdir(pdb_path)

        # make sure that the dir containing the PSSMs exists
        if not os.path.isdir(pssm_path):
            raise NotADirectoryError(pssm_path + ' is not a directory')
        # else:
        #     pssm_files = os.listdir(pssm_path)

        # create the outdir if necessary
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # check if we want to select a subset of PDBs
        if select is not None:
            if not os.path.isfile(select):
                raise FileNotFoundError(select + ' is not a file')
            else:
                with open(select,'r') as f:
                    select = f.readlines()
        else:
            select = None

        # get the list of PDB names
        pdbs = list(filter(lambda x: x.endswith('.pdb'),os.listdir(pdb_path)))
        if select is not None:
            pdbs = list(filter(lambda x: x.startswith(select),pdbs))

        # create the output file
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # create the partial list
        pdbs = [pdbs[i::size] for i in range(size)]
        local_pdbs = pdbs[0]

        # send the list
        for iP in range(1,size):
            mpi_comm.send(pdbs[iP],dest=iP,tag=11)

    else:
        local_pdbs = mpi_comm.recv(source=0,tag=11)

    # get all the pssm files
    all_pssm_files = os.listdir(pssm_path)

    # loop over all the PDBs
    for name in local_pdbs:

        print('[%03d] -- Creating graph of PDB %s' %(rank,name))

        # pdb name
        pdbfile = os.path.join(pdb_path,name)

        # mol name and base name
        mol_name = os.path.splitext(name)[0]

        # get the pssm files
        mol_pssm = list(filter(lambda x : x.startswith(mol_name+'.') and x.endswith('.pdb.pssm'),all_pssm_files))
        if len(mol_pssm) == 0:
            print('--> Assuming global naming scheme for pssm files')
            mol_pssm = list(filter(lambda x : x.startswith(mol_name.split('_')[0]) and x.endswith('.pdb.pssm'),all_pssm_files))

        mol_pssm.sort()
        chain_label = [m.split('.')[-3] for m in mol_pssm]


        # print the pssm and chains found
        print(' --> Found the following chains and PSSM')
        pssm = dict()
        for c,f in zip(chain_label,mol_pssm):
            print('     %s : %s' %(c,f))
            pssm[c] = os.path.join(pssm_path,f)

        # append the pssm path to the files
        mol_pssm = [os.path.join(pssm_path,f) for f in mol_pssm]

        # # check if the pssms exists
        # if os.path.isfile(mol_pssm[0]) and os.path.isfile(mol_pssm[1]):
        #     pssm = {chain_label[0]:mol_pssm[0],chain_label[1]:mol_pssm[1]}
        # else:
        #     raise FileNotFoundError(mol_pssm[0] + ' or ' + mol_pssm[1] + ' not found')

        # output file
        graphfile = os.path.join(outdir+mol_name+'.pckl')

        if export_hdf5:
            h5name = 'graphs_%d.hdf5' %rank
            f5 = h5py.File(h5name,'w')
        else:
            f5 = None

        # create the graphs
        GenGraph(pdbfile,pssm,aligned=aligned,outname=graphfile,h5file=f5)

        if export_hdf5:
            f5.close()