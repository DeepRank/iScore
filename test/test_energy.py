from iScore.energetic.internal_energy import InternalEnergy
from iScore.energetic.haddock_energy import HaddockEnergy
from iScore.energetic.energy import iscore_energy
import unittest


class TestEnergetic(unittest.TestCase):
    """ Test energetic calculation."""

    def setUp(self):
        self.pdb = './graph/1ATN.pdb'

    def test_vdw_clb(self):
        E = InternalEnergy(self.pdb)

        if E.evdw != -67.86502277452256 or E.ec != 417.7057796713902:
            raise AssertionError()

class TestReadEnergy(unittest.TestCase):
    """Test the read of HADDOCK energy."""

    def setUp(self):
        self.pdb = './rank/test/pdb/1ACB_1w.pdb'
        self.pdb_fail = './rank/train/pdb/2I25.pdb'

    def test_read(self):
        e = HaddockEnergy(self.pdb)
        e.read_energies()

        if e.evdw != -26.9989 or e.ec != -256.513 or e.edesolv != 16.2592:
            raise AssertionError()

    @unittest.expectedFailure
    def test_fail(self):

        e = HaddockEnergy(self.pdb_fail)
        e.read_energies()

class TestEnergy(unittest.TestCase):
    """Test the process of the energy terms."""

    def setUp(self):
        self.pdb = './rank/test/pdb/'

    def test(self):
        iscore_energy(pdb_path=self.pdb)
