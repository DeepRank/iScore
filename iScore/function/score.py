import os
import numpy as np
import warnings
from scipy.stats import iqr

class iscore(object):

    def __init__(self,graphrank_out='GraphRank.dat', energy_out='Energy.dat',
                 weights = [-0.941,0.041,0.217,0.032], normalize_graph_rank = True):

        self.graphrank_out = graphrank_out
        self.energy_out = energy_out
        self.weights = weights
        self.normalize_graph_rank = normalize_graph_rank

        self.features = dict()
        # Read and check, remove problematic molecule
        self.read_graphrank()
        self.read_energy()
        self.check_mol_features()

        if self.normalize_graph_rank:
            self.normalize_graphrank()
        self.score()
        self.print()

    def read_energy(self):
        """Read the energy poutput file."""

        with open(self.energy_out,'r') as f:
            data = f.readlines()

        for line in data[1:]:
            mol,vdw,clb,des = line.split()

            if mol not in self.features:
                self.features[mol] = dict()

            self.features[mol]['evdw'] = float(vdw)
            self.features[mol]['ec'] = float(clb)
            self.features[mol]['edesolv'] = float(des)

    def read_graphrank(self):
        """Read the graph rank output file."""

        with open(self.graphrank_out,'r') as f:
            data = f.readlines()

        for line in data[1:]:
            l = line.split()
            mol = l[0]
            if mol not in self.features:
                self.features[mol] = dict()
            self.features[mol]['grank'] = float(l[-1])

    def normalize_graphrank(self):
        """Normalize graphRank values"""

        data = []
        mol = self.features.keys()

        for m in mol:
            data.append(self.features[m]['grank'])

        data = self._normalize(data)

        i = 0
        for m in mol:
            self.features[m]['grank'] = data[i]
            i+= 1

    def check_mol_features(self):
        """Check and remove molecules without enough features"""

        features = set(['evdw', 'ec', 'edesolv', 'grank'])
        mols = list(self.features.keys())
        for mol in mols:
            mol_feat = set(self.features[mol].keys())
            if mol_feat != features:
                del self.features[mol]
                warnings.warn(f'Molecule {mol} is deleted from scoring due to lacking enough features')

    @staticmethod
    def _normalize(data):
        data = np.array(data)
        return (data-np.median(data))/iqr(data)

    def score(self):
        """compute and output the iScore."""

        for mol,feat in self.features.items():

            if all( k in feat for k in ['grank','evdw','ec','edesolv']):
                data = [feat['grank'],feat['evdw'],feat['ec'],feat['edesolv']]

                self.features[mol]['iscore'] = self._scoring_function(data,self.weights)
            else:
                print('Data missing for mol %s. Molecule excluded from scoring' %mol)

    def print(self):
        """Print the energy terms."""

        fname='iScorePredict.dat'
        f = open(fname,'w')
        if self.normalize_graph_rank:
            f.write('{:10} {:>14}     {:>14}     {:>14}     {:>14}     {:>14}\n'.format('#Name','nGraphRank','nEVDW','nEC','nEDESOLV','iScore'))
        else:
            f.write('{:10} {:>14}     {:>14}     {:>14}     {:>14}     {:>14}\n'.format('#Name','GraphRank','nEVDW','nEC','nEDESOLV','iScore'))
        for name,feat in  self.features.items():
            st = "{:10} {: 14.3f}     {: 14.3f}     {: 14.3f}     {: 14.3f}     {: 14.3f}\n"
            if all( k in feat for k in ['grank','evdw','ec','edesolv','iscore']):
                f.write(st.format(name,feat['grank'],feat['evdw'],feat['ec'],feat['edesolv'],feat['iscore']))
        f.close()

    @staticmethod
    def _scoring_function(features,weights):

        if isinstance(weights,list):
            weights = np.array(weights)
        if isinstance(features,list):
            features = np.array(features)

        return np.sum(features*weights)
