from iScore.energetic.energetic import EnergeticTerms
import unittest


class TestEnergetic(unittest.TestCase):
    """ Test energetic calculation."""

    def setUp(self):
        self.pdb = pdb = './graph/1ATN.pdb'

    def test_vdw_clb(self):
        E = EnergeticTerms(self.pdb)

        if E.evdw != -67.86502277452256 or E.ec != 417.7057796713902:
            raise AssertionError()
