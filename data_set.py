import sys, os
import numpy as np

class data:
	data_ran = np.random.random((3, 3))

	#dt_F0 = np.dtype([('F0', float)])
	dt_F0 = np.zeros((4, 4))
	dt_F1 = np.zeros((4, 4))
	dt_F2 = np.zeros((4, 4))
	dt_F3 = np.zeros((4, 4))
	dt_F4 = np.zeros((4, 4))

	dt_FF = np.zeros((4, 1))


	def __init__(self):
		print('Intializing Data.......')
		self.dt_F0 = np.matrix([[0.50, 0.55, 0.33, 2.38], [0.55, 0.18, -1.18, -0.40], [0.33, -1.18, -0.94, 1.46], [2.38, -0.40, 1.46, 0.17]])
		self.dt_F1 = np.matrix([[5.19, 1.54]])

	def F0(self):
		print(self.dt_F0)

