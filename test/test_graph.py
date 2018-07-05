from iScore.graph import GenGraph, Graph
import unittest

class TestGraph(unittest.TestCase):
	"""Test Graph generation."""

	def test_graph(self):

		g = GenGraph(self.pdb,self.pssm,export=True,outname=self.pkl)

		g = Graph(self.pkl)
		gcheck = Graph(self.check)
		check = g.compare(gcheck)

		if not check:
			raise AssertionError()

	def setUp(self):
		self.pdb = './graph/1ATN.pdb'
		self.pkl = './graph/1ATN.pckl'
		self.pssm = {'A':'./graph/1ATN.A.pdb.pssm','B':'./graph/1ATN.B.pdb.pssm'}
		self.check = './graph/1ATN.mat'


if __name__ == '__main__':
    unittest.main()




