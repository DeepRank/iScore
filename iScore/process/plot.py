import numpy as np
from collections import OrderedDict as odict

class Plot(object):

    def __init__(self):
        """Plot Data (hitrate) generated with iScore."""

        self.available_scores = []
        self.data = odict()

    def read_iscore_file(self,fname):
        """Read the utput file of a iScore calculation.

        Args:
            fname (str): file name containing the data
        """

        with open(fname,'r') as f:
            data = f.readlines()

        for il, l in enumerate(data):
            l = l.split()

            if il == 0:
                keys = l[1:]
                self.available_scores += keys
            else:
                mol_name = l[0]

                if mol_name not in self.data:
                    self.data[mol_name] = dict()

                for k,v in zip(keys,l[1:]):
                    self.data[mol_name][k] = float(v)

    def read_capri(self,fname):
        """Read the utput file of a capri calculation.

        Args:
            fname (str): file name containing the data
        """

        with open(fname,'r') as f:
            data = f.readlines()

        for l in data:
            l = l.split()
            mol_name = l[0]
            cat = int(l[1])

            if mol_name not in self.data:
                self.data[mol_name] = dict()
            self.data[mol_name]['capri'] = cat
        self.available_scores.append('capri')

    def read_haddock(self,fname):
        """Read the utput file of a haddock calculation.

        Args:
            fname (str): file name containing the data
        """

        with open(fname,'r') as f:
            data = f.readlines()

        for l in data:
            l = l.split()
            mol_name = l[0]
            hs = float(l[1])

            if mol_name not in self.data:
                self.data[mol_name] = dict()
            self.data[mol_name]['haddock'] = hs

        self.available_scores.append('haddock')

    def get_hitrate(self,data='iScore',ref='capri',normalize=True):
        """Compute the hitrate of a given scoring.
        Args:
            data (str): name of the data to process
            ref (str): name of the data used as ground truth
        """

        if data not in self.available_scores:
            raise ValueError('%s not in available socres : ' %data, self.available_scores)

        values, truth = [], []
        for k,v in self.data.items():

            if all(k in v for k in [data,ref]):
                values.append(v[data])
                truth.append(v[ref])

        # get the data
        values = np.array(values)
        truth = np.array(truth)

        # sort the data
        index = np.argsort(values)

        # revert if necessary
        if data.endswith("GraphRank"):
              index = index[::-1]


        #sort the data
        values = values[index]
        truth = truth[index]

        # return the hitrate
        if ref == 'capri':
            hr =  np.cumsum(truth<=3)
            if normalize:
                return hr/np.max(hr)
            else:
                return hr
        else:
            raise ValueError('Only possible reference is CAPRI for the moment')


    def get_score(self,name):

        names = self.data.keys()
        score = []
        for n in names:
            score.append(self.data[n][name])
            
        return np.array(score)