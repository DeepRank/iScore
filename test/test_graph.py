from iScore.graphrank.graph import GenGraph, Graph
import unittest

class TestGraph(unittest.TestCase):
    """Test Graph generation."""

    @staticmethod
    def assert_graph(pdb,pssm,pkl,check,isaligned):

        g = GenGraph(pdb,pssm,aligned=isaligned,export=True,outname=pkl)

        g = Graph(pkl)
        gcheck = Graph(check)
        check = g.compare(gcheck)

        if not check:
            raise AssertionError()

    def test_aligned(self):
        self.assert_graph(self.a_pdb,self.a_pssm,self.a_pkl,self.a_check,isaligned=True)

    def test_notalligned(self):
        self.assert_graph(self.na_pdb,self.na_pssm,self.na_pkl,self.na_check,isaligned=False)

    @unittest.expectedFailure
    def test_notalligned_fail(self):
        self.assert_graph(self.na_pdb,self.na_pssm,self.na_pkl,self.na_check,isaligned=True)

    def setUp(self):

        self.a_pdb = './graph/1ATN.pdb'
        self.a_pkl = './graph/1ATN.pckl'
        self.a_pssm = {'A':'./graph/1ATN.A.pdb.pssm','B':'./graph/1ATN.B.pdb.pssm'}
        self.a_check = './graph/1ATN.mat'


        self.na_pdb = './graph_notalligned/1ATN.pdb'
        self.na_pkl = './graph_notalligned/1ATN.pckl'
        self.na_pssm = {'A':'./graph_notalligned/1ATN.A.pssm','B':'./graph_notalligned/1ATN.B.pssm'}
        self.na_check = './graph_notalligned/1ATN.mat'

if __name__ == '__main__':
    unittest.main()
