import os, sys
import numpy as np
from scipy.stats import iqr
from iScore.energetic.haddock_energy import HaddockEnergy

class iscore_energy(object):

    def __init__(self,pdb_path='./pdb',method='haddock'):

        self.pdb_path=pdb_path
        self.method=method

        if method is not None:
            self.get_energies()
            self.normalize_features()
            self.print()

    def get_energies(self):
        """Get the enegetic terms with the spcified method."""

        self.mol_name = []
        self.evdw = []
        self.ec = []
        self.edesolv = []

        if not os.path.isdir(self.pdb_path):
            raise NotADirectoryError()

        filenames = list(filter(lambda x: x.endswith('.pdb'),os.listdir(self.pdb_path)))

        try:

            for f in filenames:

                self.mol_name.append(os.path.splitext(f)[0])
                if self.method == 'haddock':

                    e = HaddockEnergy(os.path.join(self.pdb_path,f))
                    e.read_energies()

                    self.evdw.append(e.evdw)
                    self.ec.append(e.ec)
                    self.edesolv.append(e.edesolv)

                else:

                    raise ValueError('Only Haddock energy term supported so far')

            self.evdw = np.array(self.evdw)
            self.ec = np.array(self.ec)
            self.edesolv = np.array(self.edesolv)

        except Exception as e:

            print(e)
            print(" Warning : Issue encountered during the calculation of the energy terms")
            print("           All energy terms set to 0 (i.e. GraphRank score only)")

            nmol = len(filenames)
            self.evdw = np.zeros(nmol)
            self.ec = np.zeros(nmol)
            self.edesolv = np.zeros(nmol)

    def normalize_features(self):
        """Normalize the feature."""

        self.evdw = self._normalize(self.evdw)
        self.ec = self._normalize(self.ec)
        self.edesolv = self._normalize(self.edesolv)


    @staticmethod
    def _normalize(x):
        """Normalizing function."""
        N = iqr(x)
        if N != 0.:
            return (x-np.median(x))/iqr(x)
        else :
            return x

    def print(self):
        """Print the energy terms."""

        fname='Energy.dat'
        f = open(fname,'w')
        f.write('{:10} {:>14}     {:>14}     {:>14}\n'.format('#Name','nEVDW','nEC','nEDESOLV'))
        for name,evdw,ec,edesol in zip(self.mol_name,self.evdw,self.ec,self.edesolv):
            st = "{:10} {: 14.3f}     {: 14.3f}     {: 14.3f}\n"
            f.write(st.format(name,evdw,ec,edesol))
        f.close()

if __name__ == '__main__':
    pdbs = '../../example/training_set/test/pdb/'
    e = EnergeticTerms(pdbs)
    e.get_energies()
    e.normalize_features()
    e.print()