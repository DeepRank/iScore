from iScore.function.score import iscore
import unittest


class TestScore(unittest.TestCase):
    """ Test scoring calculation."""

    def setUp(self):
        self.energy = './score/Energy.dat'
        self.graph = './score/GraphRank.dat'

    def test(self):
        iscore(graphrank_out=self.graph,energy_out=self.energy)