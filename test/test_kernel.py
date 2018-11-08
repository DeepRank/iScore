from iScore.graphrank.graph import iscore_graph
from iScore.graphrank.kernel import Kernel
import numpy as np
import unittest

class TestKernel(unittest.TestCase):
	"""Test the kernels."""

	def test_kernel(self):


		#init and load the data
		ker = Kernel(testIDs='./kernel/testID.lst',trainIDs='./kernel/trainID.lst',
			         test_graph='./kernel/graph/',
			         train_graph='./kernel/graph/')
		ker.import_from_mat()

		# get the path of the check file
		checkfile = ker.get_check_file(fname='./kernel/check/K.mat')

		# run the calculations
		check_values = ker.run(lamb=1.0,walk=4,check=checkfile)

		if not np.all(check_values):
			raise AssertionError()

	def setUp(self):
		# create all the graphs of the pdb in ./pdb/
		iscore_graph(pdb_path='./kernel/pdb/',pssm_path='./kernel/pssm/',outdir='./kernel/graph/')


if __name__ == '__main__':
	unittest.main()