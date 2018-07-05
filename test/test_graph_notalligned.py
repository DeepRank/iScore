from iScore.graph import GenGraph, Graph
import unittest

class TestGraphNotAlligned(unittest.TestCase):
	"""Test Graph generation."""

	def test_graph(self):

		g = GenGraph(self.pdb,self.pssm,aligned=False,export=True,outname=self.pkl)

		g = Graph(self.pkl)
		gcheck = Graph(self.check)
		check = g.compare(gcheck)

		if not check:
			raise AssertionError()

	def setUp(self):
		self.pdb = './graph_notalligned/1ATN.pdb'
		self.pkl = './graph_notalligned/1ATN.pckl'
		self.pssm = {'A':'./graph_notalligned/1ATN.A.pssm','B':'./graph_notalligned/1ATN.B.pssm'}
		self.check = './graph_notalligned/1ATN.mat'


if __name__ == '__main__':
    unittest.main()




