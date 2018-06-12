import os
from pdb2sql.pdb2sql import pdb2sql
from pdb2sql.interface import interface
from Bio import pairwise2
import numpy as np
import pickle
from .graph import Graph

class GenGraph():

    def __init__(self,pdbfile,pssmfile,aligned=True,outname=None,cutoff=6.0):

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
        for chain in ['A','B']:
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
        self.export_graph(outname)


    def check_pssm_format(self,aligned):
        if aligned:
            return len(self.pssm['A'][0]) == 25
        else:
            return len(self.pssm['A'][0]) == 48


    def read_PSSM_data(self,fname):
        """Read the PSSM data."""

        f = open(fname,'r')
        data = f.readlines()
        f.close()

        filters = (lambda x: len(x.split())>0, lambda x: x.split()[0].isdigit())
        return list(map(lambda x: x.split(),list(filter(lambda x: all(f(x) for f in filters), data))))

    def get_aligned_pssm(self):

        self._align_sequences()

        self.aligned_pssm = {}
        self.aligned_ic = {}
        for chain in ['A','B']:

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

        self.seq_aligned = {'pdb':{},'pssm':{}}
        for chain in ['A','B']:
            pdb_seq = self._get_sequence(chain=chain)
            pssm_seq = ''.join( [data[1] for data in self.pssm[chain] ] )
            self.seq_aligned['pdb'][chain], self.seq_aligned['pssm'][chain] = self._get_aligned_seq(pdb_seq,pssm_seq)


    def _get_sequence(self,chain='A'):
        if name in resmap_inv.keys():
            data = [(numb,self.resmap_inv[name]) for numb,name in self.pdb.get('resSeq,resName',chainID=chain)]
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

        self.aligned_pssm, self.aligned_ic = {},{}
        for chain in ['A','B']:
            for l in self.pssm[chain]:
                resi = int(l[0])
                resn = self.resmap[l[1]]
                self.aligned_pssm[(chain,resi,resn)] = l[4:24]
                self.aligned_ic[(chain,resi,resn)] = l[24]



    def construct_graph(self,verbose=False,print_res_pairs=False):

        db = interface(self.pdbfile)
        res_contact_pairs = db.get_contact_residues(cutoff = self.cutoff, return_contact_pairs=True)

        # remove the non residues
        for res in res_contact_pairs.keys():
            if res[2] not in self.resmap_inv:
                res_contact_pairs.pop(res,None)
                Warning('--> Residue ',res,' not valid')

        # remove the ones that are not in PSSM
        for res in list(res_contact_pairs.keys()):
            if res not in self.aligned_pssm:
                res_contact_pairs.pop(res,None)
                Warning('--> Residue ',res,' not found in PSSM file')

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
                    if verbose:
                        print('removed node', res)
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

    def export_graph(self,fname):

        nodes_data,nodes_info = [],[]
        for res in self.nodes:

            pssm = self.aligned_pssm[res]
            nodes_data.append(pssm)

            ic = self.aligned_ic[res]
            nodes_info.append(ic)

        nodes_data = np.array(nodes_data)
        nodes_info = np.array(nodes_info)

        name = os.path.splitext(fname)[0]
        nodes_pssm = np.array(nodes_data).astype('int')
        nodes_info = np.array(nodes_info).astype('float')

        edges = np.array(self.edges)

        graph = Graph(name = name,
                      nodes_pssm = nodes_pssm,
                      nodes_info = nodes_info,
                      edges_index = edges)

        graph.pickle(fname)

# if __name__ == "__main__":

#     pdb = './example_input/1E96_1w.pdb'
#     pssm = {'A':'./example_input/1E96.A.pssm','B':'./example_input/1E96.B.pssm'}

#     g = graphCreate(pdb,pssm)
#     g.get_aligned_pssm()

#     for chain in ['A','B']:
#         for x,y in zip(g.seq_aligned['pdb'][chain],g.seq_aligned['pssm'][chain]):
#             print(chain,x,y)

#     g.construct_graph()
#     g.export_graph('test.pkl')