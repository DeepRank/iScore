import os, re
import numpy as np


class HaddockEnergy(object):

    def __init__(self,filename):
        self.filename = filename

    def read_energies(self):
        """ Read the energy terms of an HADDOCK pdb file."""

        energy_line = self._get_line('energies').split(',')
        self.evdw = float(energy_line[5])
        self.ec   = float(energy_line[6])

        desolv_line = self._get_line('Desolvation').split()
        self.edesolv = float(desolv_line[-1])

    def _get_line(self,keyword,first_occurence_only=True):
        """ Returns a line (or all lines) containing the keyword.

        Args:
            keyword (str) : word that must be in line
            first_occurence_only (bool, optional) : return only the first occurence if True
        """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(self.filename)

        all_lines = []
        with open(self.filename,'r') as f:
            for line in f:
                if re.search(keyword,line):
                    if first_occurence_only:
                        return line
                    else:
                        all_lines.append(line)

        if (not first_occurence_only and all_lines == 0) or first_occurence_only:
            raise ValueError('Keyword %s not found in %s\n            Make sure that %s was generated with HADDOCK' \
                              %(keyword,self.filename,self.filename))
            return None
        else:
            return all_lines