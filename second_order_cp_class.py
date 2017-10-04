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
#from scipy import special, optimize
from numpy.linalg import inv
from numpy import linalg as LA

#*************************************************************************
# This is our class with the data elements that will be used
#*************************************************************************
from data_set import data

#*************************************************************************
# Print out the paths available to this program during run time. Uncomment
# if there is a need to debug
#*************************************************************************
print('\n'.join(sys.path))


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
	#	X = diag{X1,...,Xq}
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
		# Creeaate and initial with zeros matrices X and S
		#*********************************************************
		X = np.zeros((m,m));
		S = np.zeros((m,m));

		# used in the loop
		ct = 0;

		#*********************************************************
		# For loop to iterate over each of the element in n
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

			xi = x[ct:ct+ni]

			a = xi[1:ni];

			b = xi[0]*Ii;

			ab = np.hstack((a, b));

			Xi = np.vstack((xi.reshape(xi.shape[1], xi.shape[0]), ab));

			X[(ct):(ct+ni), (ct):(ct+ni)] = Xi;

			si = s[ct:(ct+ni)];

			a = si[1:ni]

			b = si[0]*Ii

			ab = np.hstack((a,b));

			Si = np.vstack((si.reshape(si.shape[1], si.shape[0]), ab));

			S[(ct):(ct+ni), (ct):(ct+ni)] = Si;

			ct=ct+ni

		return X, S


	act = 0
	def find_alpha(self, x, dx, n):
		q = n.shape[1];

		aw = np.zeros((q,1));

		act = 0;

		for i in range(0, q):
			aw1=0
			aw2=0

			ni = n.item(i);

			xi = x[(act):(act+ni)];

			dxi = dx[(act):(act+ni)];

			x1 = xi[0];

			xr = xi[1:ni];

			d1 = dxi[0];

			dr = dxi[1:ni];

			p0_1 = np.around(math.pow(d1, 2), decimals=4);
			p0_2_0 = np.around(LA.norm(dr), decimals=4);
			p0_2_1 = np.around(math.pow(p0_2_0, 2), decimals=4);

			p0 = np.around((p0_1 - p0_2_1), decimals=4);


			p1 = ((2*(x1*d1 - np.dot(np.transpose(xr), dr)))[0])[0];

			p2 = math.pow(x1, 2) - math.pow((LA.norm(xr)), 2);

			if d1 >= 0:
				aw1 = 1;
			else: 
				aw1 = 0.99*(x1/((-1)*d1));
			

			p_stack	= np.hstack((p0, p1));
			p_stack	= np.hstack((p_stack, p2));

			rt = (np.sort(np.roots(p_stack))).reshape(2,1);

			if p0 > 0 and rt[0] > 0:
				aw2 = rt[0];
			elif p0 > 0 and rt[1] < 0:
				aw2 = 1;
			elif p0 < 0:
				aw2 = rt[1];

			aw_stack = np.hstack((aw1, aw2));

			aw[i] = min(aw_stack);

			act = act + ni;

		a = np.around(min(aw), decimals=4);

		return a


	#*****************************************************************
	# The Second Order Cone Proramming function
	#*****************************************************************
	def socp(self) :

		#********************************************************
		# Get q, the number of elements in n i.e. 
		#	n of i = 1, 2, ..., q
		#********************************************************
		q = (self.data_obj.n).shape[1];

		#********************************************************
		# Sum of the elements in n ????? WHY
		#********************************************************
		m = np.sum(self.data_obj.n);

		#********************************************************
		#
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

		x = self.data_obj.x0;
		s = self.data_obj.s0;
		y = self.data_obj.y0;

		#********************************************************
		# Calculate the primal dual gap of the ith
		#
		#	minimize the gap?????? see ex 14.5
		#********************************************************
		gap = np.sum(x*s);

		#********************************************************
		# The mean value of the gap
		#********************************************************
		mu = gap/q;

		#********************************************************
		# Number of elements in b
		# Will be used in the while loop because ????????
		#********************************************************
		nb = (self.data_obj.b).shape[0];

		k = 0;


		while(mu >=  self.data_obj.epsi):
			#*************************************************
			# Call the get_xs function to get the next X S
			#*************************************************
			X, S = self.get_xs(x, s, self.data_obj.n)

			sm = 2*m + nb;

			M = np.zeros((sm,sm));

			M[0:nb, 0:m] = self.data_obj.A;

			M[(nb):(nb+m), (m):m2] = Im;

			M[(nb):(nb+m),(m2):sm] = np.transpose(self.data_obj.A);

			M[(nb+m):sm, 0:m2] = np.around(np.column_stack((S, X)), decimals=4);

			bw_0 = self.data_obj.b-np.dot(self.data_obj.A, x);
			bw_1 = self.data_obj.c - s - np.dot( np.transpose(self.data_obj.A), y )
			bw_2 = self.data_obj.sig*mu*e-X*s
			bw_2 = (bw_2.diagonal()).reshape((bw_2).shape[0], 1)

			bw = np.vstack((bw_0,bw_1));
			bw = np.around(np.vstack((bw,bw_2)), decimals=4);

			delt = np.around(np.dot(inv(M), bw), decimals=4);

			dx = delt[0:m];

			ds = delt[m:m2];

			dy = delt[m2:sm];

			a1 = self.find_alpha(x, dx, self.data_obj.n);

			a2 = self.find_alpha(s, ds, self.data_obj.n);

			t = np.around(self.data_obj.c - np.dot(np.transpose(self.data_obj.A),y), decimals=4);

			dt = np.around(np.dot(np.transpose(-1*self.data_obj.A), dy), decimals=4);

			a3 = self.find_alpha(t, dt, self.data_obj.n);

			a_stack = np.hstack((a1, a2));
			a_stack = np.hstack((a_stack, a3));

			a = np.around(0.5*min(a_stack), decimals=4);

			x = np.around(x + (a*dx), decimals=4);

			s = np.around(s + (a*ds), decimals=4);

			y = np.around(y + (a*dy), decimals=4);

			k = k + 1;

			gap = np.around(((np.dot(np.transpose(x), s))[0])[0], decimals=4);

			mu = np.around(gap/q, decimals=4);

		fs = ((np.dot(np.transpose(self.data_obj.c), x))[0])[0];
		print("fs: ", fs)


