from iScore.process import DataProcess
import matplotlib.pyplot as plt
import unittest

class TestProcess(unittest.TestCase):
    """Test Post Process of data."""

    def test_hitrate(self):

        proc = DataProcess(self.output)
        proc.add_label(self.caseID)
        proc.hit_rate(showfig=False)


    def setUp(self):
        self.output = './process/iScorePredict.dat'
        self.caseID = './process/caseID.lst'