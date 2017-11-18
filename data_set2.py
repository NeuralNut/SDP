##########################################################################
# Authors: Richard Qualis,
# data_set.py
#
# I will add comments soon
#
##########################################################################

#*************************************************************************
# Import python libraries that that contains the tools to be used
#*************************************************************************
import sys, os
import numpy as np


#*************************************************************************
# Start of the the python class
#*************************************************************************
class data:

	Q_M = (np.array([[4, 7], [2, 6]])).reshape(2,2)
	Q_M = (np.array([[3, 0 ,2], [2, 0, -2], [0, 1, 1]])).reshape(3,3)

	#*****************************************************************
	# The input data set
	#
	#	These are the initial data set parameters (A, b, c)
	#*****************************************************************
	A = np.array([[ 1, 0, 0, 0, 0, 0, 0, 0, 0], \
			[0, 1, 0, 0, 0.5, 0, 0, 0, 0], \
			[0, 0, 1, 0, 0, 1, 0, 0, 0], \
			[0, -1, 0, 0, 0, 0, 0, 1, 0], \
			[0, 0, -1, 0, 0, 0, 0, 0, 1]])

	b = (np.array([1, 0, 0, 0, 0])).reshape(5,1)

	c = (np.array([0, 0, 0, 1, 0, 0, 1.5, -2.8, -3])).reshape(9,1);

	#*****************************************************************
	# Initial vector sets
	#*****************************************************************
	#	These initial vector set (x0, s0, y0)
	#
	#	x0 is an element in the strictly feasible set of primal
	#	(s0, y0) is an element in the feasible set of dual
	#
	#*****************************************************************
	x0 = (np.array([1, 0, 0, 0.1, 0, 0, 0.1, 0, 0])).reshape(9,1);

	s0 = (np.array([4.1, -2.8, -3, 1, 0, 0, 1.5, 0, 0])).reshape(9,1);

	y0 = (np.array([-4.1, 0, 0, -2.8, -3])).reshape(5,1);

	# NEED TO UNDERSTAND WHAT IS THIS n FOR and why. It is always used 
	# to determine q and the same values used for each interior point
	# generated
	n = (np.array([3, 3, 3])).reshape(1,3);

	# NEED TO UNDERSTAND WHY THIS
	sig = 1e-5;
	epsi = 1e-6;

	# sig = .5;
	# epsi = .4;

	#*****************************************************************
	# Python initialization method. It is executed each time this class
	# is used, by default
	#*****************************************************************
	def __init__(self):
		print('Intializing Data.......')

