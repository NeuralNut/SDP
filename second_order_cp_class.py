##########################################################################
# Authors: Richard Qualis,
# second_order_cp_class.py
#
# I will add comments soon
#
##########################################################################

#*************************************************************************
# Import python libraries that that contains the tools to be used
#*************************************************************************
import sys, os
import numpy as np
import math

from scipy.sparse import csr_matrix
from numpy.linalg import inv
from numpy import linalg as LA

#*************************************************************************
# This is our class with the data elements that will be used
#*************************************************************************
from data_set import data

#*************************************************************************
# Start of the the python class
#*************************************************************************
class second_order_cp:
	data_obj = data()

	#*****************************************************************
	# Perform initialization
	#*****************************************************************
	def __init__(self):
		print('Intializing SOCP.......')

	#*****************************************************************
	# get_xs function:
	#   Using the current x and s of k, and n construct matrices  
	#   X and S as defined in equations:   14.125d and 14.125e	
	#	
	#	X = diag{X1,...,Xq}
	#
	#*****************************************************************
	def get_xs(self, x, s, n):
		#*********************************************************
		# Determine the number of elements, q, in n for this ith 
		# interior point. Should always be the same since vector
		# n will always be the same.
		#*********************************************************
		m = np.sum(n);

		q = n.shape[1];

		#*********************************************************
		# Create and initialize with zeros matrices X and S
		#*********************************************************
		X = np.zeros((m,m));
		S = np.zeros((m,m));

		# used in the loop
		ct = 0;

		#*********************************************************
		# For loop to iterate over each of the element in n and
		# generate the X and S marices.
		#*********************************************************
		for i in range(0, q):
			ni = n.item(i);

			#************************************************
			# Create an identity matrix of with size equaling 
			# the ith element of n minus 1
			#
			# If n of i = 5, then
			#
			#      |1 0 0 0|
			#  I = |0 1 0 0|
			#      |0 0 1 0|
			#      |0 0 0 1| 
			#
			#************************************************
			Ii = np.identity(ni-1)

			#************************************************
			# Get from the first element in this x vector to 
			# the n_i^th element. So using the sample data, 
			# since n = [5, 3, 3], the for the first
			# iteration, q=0, extract the fist 5 elements 
			# from this x which will be asigned to xi. This
			# will result in a 5 row vector.
			#
			# Using the sample data, the firt iteration would
			# be x = [1, 0, 0, 0, 0, 0.1, 0, 0, 0.1, 0, 0]^T
			# hence xi = [1, 0, 0, 0, 0]^T
			#
			# For q=1, xi = [0.1, 0, 0]^T
			# For q=2, xi = [0.1, 0, 0]^T
			#************************************************
			xi = x[ct:ct+ni]

			#************************************************
			# From the xi vector created above, get the all
			# elements, except the fist. This is the same as
			# the element in the ni-1 element, but the minus
			# element is the first element in the vector
			#
			# Using the sample data, the firt iteration would
			# be a = [0, 0, 0, 0]^T taken from 
			# xi = [1, 0, 0, 0, 0]^T
			#
			# WHY?????????
			#************************************************
			a = xi[1:ni];

			#************************************************
			# Multiply the identity matrix by the first element
			# in the xi vector.  This will result in a matrix
			# of dimention ni-1 by ni-1. Using the sample
			# data and the first iteration, this will be:
			#
			#	        |1 0 0 0|
			#    b = 1*Ii = |0 1 0 0|
			#	        |0 0 1 0|
			#	        |0 0 0 1|
			#
			# It is by coincidence that all the diagonals are
			# 1, but this will not be the case for subsequent
			# iteratins. This changes based on the value.
			# This will be evident when creating the S matrix.
			#************************************************
			b = xi[0]*Ii;

			#************************************************
			# Aggregate the vector a and the matrix b, 
			# horizontally. So for the first iteration, this
			# will be:
			#
			#	 |0 1 0 0 0|
			#   ab = |0 0 1 0 0|
			#	 |0 0 0 1 0|
			#	 |0 0 0 0 1|
			#
			#************************************************
			ab = np.hstack((a, b));

			#************************************************
			# Transpose the xi vector to a row vector and then
			# aggregate it with the ab matrix created above.
			#
			#	 |1 0 0 0 0|
			#	 |0 1 0 0 0|
			#   Xi = |0 0 1 0 0|
			#	 |0 0 0 1 0|
			#	 |0 0 0 0 1|
			#
			#************************************************
			Xi = np.vstack((xi.reshape(xi.shape[1], xi.shape[0]), ab));

			#************************************************
			# Now, insert the Xi matrix into the X matrix in
			# the position defined by the dimension of the
			# matrix (horizontally stacking):
			#
			# Using the sample data, the first iteration would
			# be X[(ct):(ct+ni), (ct):(ct+ni)] which is
			# [0:0+5, 0:0+5] = [0:5, 0:5]
			#
			# When all is done, all the Xi's created will be
			# inserted into this X matrix
			#************************************************
			X[(ct):(ct+ni), (ct):(ct+ni)] = Xi;




			#************************************************
			# Do the sameting that was done for the X matrix
			# for the S matrix, but using the s matrix 
			# elements.
			#************************************************
			si = s[ct:(ct+ni)];

			a = si[1:ni]

			b = si[0]*Ii

			ab = np.hstack((a,b));

			Si = np.vstack((si.reshape(si.shape[1], si.shape[0]), ab));

			S[(ct):(ct+ni), (ct):(ct+ni)] = Si;


			ct=ct+ni

		#********************************************************
		# Return both matrices X and S
		#********************************************************
		return X, S



	#********************************************************
	# computes alpha_k by using Eq. (14.126)
	#
	# x: current point xk (or sk)
	# dx: current increment dxk (or dsk)
	# n: n = [n1 n2 ... nq], see Eq. (14.101)
	#
	# Will return 
	#	a: value of alpha obtained
	#********************************************************
	def find_alpha(self, x, dx, n):

		#********************************************************
		# Determine the number of elements in n
		#********************************************************
		q = n.shape[1];

		#********************************************************
		# Initialize aw a q * 1 column vector
		#********************************************************
		aw = np.zeros((q,1));

		act = 0;

		#********************************************************
		# For the number of elements in n, search the path to
		# determine the alpha_k
		#********************************************************
		for i in range(0, q):
			aw1=0
			aw2=0

			#********************************************************
			# get the ith element from n
			#********************************************************
			ni = n.item(i);

			#********************************************************
			# Get xi. For this data set and the first iteration
			# x[(act):(act+ni)] = x[0:0+5] = x[0:5]
			#		    = [1, 0, 0, 0, 0]^T
			#********************************************************
			xi = x[(act):(act+ni)];

			#********************************************************
			# Get dxi. For this data set and the first iteration
			# dx[(act):(act+ni)] = dx[0:0+5] = dx[0:5]
			#		     = [0, -0.0487, 0.1202, 0, 0]
			#********************************************************
			dxi = dx[(act):(act+ni)];

			#********************************************************
			# Get the first element of xi
			#********************************************************
			x1 = xi[0];

			#********************************************************
			# Get elements from xi[1:5], for this sample and first
			# iteration
			#********************************************************
			xr = xi[1:ni];

			#********************************************************
			#  Get the first element from dxi
			#********************************************************
			d1 = dxi[0];

			#********************************************************
			# Get elements from dxi[1:5], for this sample and first
			# iteration.
			#********************************************************
			dr = dxi[1:ni];

			#********************************************************
			# Calculate p0
			#********************************************************
			p0_1 = np.around(math.pow(d1, 2), decimals=4);
			p0_2_0 = np.around(LA.norm(dr), decimals=4);
			p0_2_1 = np.around(math.pow(p0_2_0, 2), decimals=4);

			p0 = np.around((p0_1 - p0_2_1), decimals=4);
			p1 = ((2*(x1*d1 - np.dot(np.transpose(xr), dr)))[0])[0];
			p2 = math.pow(x1, 2) - math.pow((LA.norm(xr)), 2);

			#********************************************************
			# Determine aw1
			#********************************************************
			if d1 >= 0:
				aw1 = 1;
			else: 
				aw1 = 0.99*(x1/((-1)*d1));
			

			p_stack	= np.hstack((p0, p1));
			p_stack	= np.hstack((p_stack, p2));

			#*************************************************
			# Calc roots of the polynomial represented by p 
			# as a column vector.
			#*************************************************
			rt = (np.sort(np.roots(p_stack))).reshape(2,1);

			#*************************************************
			# Determine aw2
			#*************************************************
			if p0 > 0 and rt[0] > 0:
				aw2 = rt[0];
			elif p0 > 0 and rt[1] < 0:
				aw2 = 1;
			elif p0 < 0:
				aw2 = rt[1];


			aw_stack = np.hstack((aw1, aw2));



			aw[i] = min(aw_stack);

			act = act + ni;

		#*************************************************
		# Return the minimum of the 2 aw's
		#*************************************************
		a = np.around(min(aw), decimals=5);

		return a





	# SOCP
	#*****************************************************************
	# The Second Order Cone Proramming function. This is the main
	# entry point of the program.
	#*****************************************************************
	def socp(self) :
		#********************************************************
		# Get q, based on the number of elements in n i.e. 
		#	n of i = 1, 2, ..., q
		#********************************************************
		q = (self.data_obj.n).shape[1];

		#********************************************************
		# Sum of the elements in n ????? WHY
		#********************************************************
		m = np.sum(self.data_obj.n);

		#********************************************************
		# Double the sum
		#********************************************************
		m2 = 2*m;

		#********************************************************
		# Create an m by m identity matrix
		#********************************************************
		Im = np.identity(m);
		
		#********************************************************
		# Create a m by 1, vector, with all 1s
		#********************************************************
		e = np.ones((m,1));

		#********************************************************
		# Get a copy of the initial x, y, s vectors
		#********************************************************
		x = self.data_obj.x0;
		s = self.data_obj.s0;
		y = self.data_obj.y0;

		#********************************************************
		# Calculate the primal dual-gap of the ith
		#
		#	minimize the gap?????? see ex 14.5
		#********************************************************
		gap = np.around(((np.dot(np.transpose(x), s))[0])[0], decimals=4);

		#********************************************************
		# The mean value of the gap wrt to q the number of 
		# elements in n
		#********************************************************
		mu = gap/q;


		#********************************************************
		# Number of elements in b
		# Will be used in the while loop because ????????
		#********************************************************
		nb = (self.data_obj.b).shape[0];

		k = 0;

		#********************************************************
		# While loop that will 
		#********************************************************
		while(mu >=  self.data_obj.epsi):
			#*************************************************
			# Solve equation 14.125 for {dx, ds, dy}
			#*************************************************

			#*************************************************
			# Use the get_xs function to calculate the next 
			# X S
			#*************************************************
			X, S = self.get_xs(x, s, self.data_obj.n)

			#*************************************************
			# Create matrix M. First determine what the size
			# should be. 
			#
			#*************************************************
			sm = 2*m + nb;

			#*************************************************
			# Create a matrix of the size and initialize it 
			# with zeros
			#*************************************************
			M = np.zeros((sm,sm));

			#*************************************************
			# Insert matrix A into the matrix M in the 
			# position denoted by 0:nb, 0:m
			#*************************************************
			M[0:nb, 0:m] = self.data_obj.A;

			#*************************************************
			# Insert the identity matrix, 11 by 11 for this 
			# sample data (5+3+3)
			#*************************************************
			M[(nb):(nb+m), (m):m2] = Im;

			#*************************************************
			# Insert the transpose of A into the matrix M
			# into the position:
			#	(nb):(nb+m),(m2):sm
			#
			# For the sample data and the first iteration
			#       = [5:5+11, 22:27] 
			#       = [6:16, 23:27]
			#*************************************************
			M[(nb):(nb+m),(m2):sm] = np.transpose(self.data_obj.A);

			#*************************************************
			# Now, first aggregate the S and X matrix then insert
			# then insert them into the matrix M into the 
			# position:
			# 	(nb+m):sm, 0:m2 = [5+11:27, 0:22]
			#			= [16:27, 0:22]
			# Next to each other
			#*************************************************
			M[(nb+m):sm, 0:m2] = np.around(np.column_stack((S, X)), decimals=4);

			#*************************************************
			# Calcualte bw:
			# 	bw = [b-A*x; c-s-A'*y; sig*mu*e-X*s]
			# Below, for clarity, they are each calculated
			# and then stacked verically to create a single
			# vector 27 rows 
			#*************************************************
			bw_0 = self.data_obj.b-np.dot(self.data_obj.A, x);
			bw_1 = self.data_obj.c - s - np.dot( np.transpose(self.data_obj.A), y )
			bw_2 = self.data_obj.sig*mu*e-X*s
			bw_2 = (bw_2.diagonal()).reshape((bw_2).shape[0], 1)

			bw = np.vstack((bw_0,bw_1));
			bw = np.around(np.vstack((bw,bw_2)), decimals=4);


			# **************************************************************************
			# Calculate the delta using the matrix M which will contain all deltas
			#
			# This is giving me a lot of different figures that causes the rest of the 
			# calculations to be thrown off
			delt = np.around(np.dot(inv(M), bw), decimals=4);
			# **************************************************************************

			#*************************************************
			# Delta x
			#*************************************************
			dx = delt[0:m];

			#*************************************************
			# Delta s
			#*************************************************
			ds = delt[m:m2];

			#*************************************************
			# Delta y
			#*************************************************
			dy = delt[m2:sm];


			#*************************************************
			# Calculate the alphas using the find_alpha method
			#
			# Perform line search using Eq. (14.126)
			#*************************************************
			a1 = self.find_alpha(x, dx, self.data_obj.n);

			a2 = self.find_alpha(s, ds, self.data_obj.n);

			t = np.around(self.data_obj.c - np.dot(np.transpose(self.data_obj.A),y), decimals=4);

			dt = np.around(np.dot(np.transpose(-1*self.data_obj.A), dy), decimals=4);

			a3 = self.find_alpha(t, dt, self.data_obj.n);


			#*************************************************
			# Get the minimum between a1, a2, and a3 and use
			# it to calculat the value of a
			#*************************************************
			a_stack = np.hstack((a1, a2));
			a_stack = np.hstack((a_stack, a3));

			a = np.around(0.5*min(a_stack), decimals=4);


			#*************************************************
			# Caluculate the new x, s, and y:
			#	vector x + (scalar a * dx)
			#*************************************************
			x = np.around(x + (a*dx), decimals=4);

			s = np.around(s + (a*ds), decimals=4);

			y = np.around(y + (a*dy), decimals=4);

			#*************************************************
			# Determine the primal-dual gap	
			#*************************************************
			gap = np.around(((np.dot(np.transpose(x), s))[0])[0], decimals=4);

			#*************************************************
			# Caluculate the mean value of the gap with respect
			# to q
			#*************************************************
			mu = np.around(gap/q, decimals=4);

			k = k + 1;

		#*************************************************
		# Now, using c and x, determine the minimimum feasible
		# value
		#*************************************************
		fs = ((np.dot(np.transpose(self.data_obj.c), x))[0])[0];
		print("fs: ", fs)

