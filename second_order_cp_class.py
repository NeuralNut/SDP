##########################################################
# Authors: Richard Qualis,
# data_set.py
#
# I will add comments soon
#
##########################################################
import sys, os
import numpy as np

from scipy.sparse import csr_matrix
from data_set import data
from numpy.linalg import inv

class second_order_cp:
	data_obj = data()

	def __init__(self):
		print('Intializing SOCP.......')


	def get_xs(self, x, s, n):
		m = np.sum(n);
		q = n.shape[1];

		X = np.zeros((m,m));
		S = np.zeros((m,m));

		ct = 0;

		for i in range(0, q):
			ni = n.item(i);

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

	def socp(self) :
		print('Executing second_order_cp......................')

		q = (self.data_obj.n).shape[1];

		m = np.sum(self.data_obj.n);

		m2 = 2*m;

		nb = (self.data_obj.b).shape[0];

		Im = np.identity(m);

		e = np.ones((m,1));

		x = self.data_obj.x0;
		s = self.data_obj.s0;
		y = self.data_obj.y0;

		gap = np.sum(x*s);

		mu = gap/q;

		k = 0;

		count = 0
		while(mu >=  self.data_obj.epsi):
			X, S = self.get_xs(x, s, self.data_obj.n)

			sm = 2*m + nb;

			M = np.zeros((sm,sm));

			M[0:nb, 0:m] = self.data_obj.A;

			M[(nb):(nb+m), (m):m2] = Im;

			M[(nb):(nb+m),(m2):sm] = np.transpose(self.data_obj.A);

			M[(nb+m):sm, 0:m2] = np.column_stack((S, X));

			bw_0 = self.data_obj.b-np.dot(self.data_obj.A, x);
			bw_1 = self.data_obj.c - s - np.dot( np.transpose(self.data_obj.A), y )
			bw_2 = self.data_obj.sig*mu*e-X*s
			bw_2 = (bw_2.diagonal()).reshape((bw_2).shape[0], 1)

			bw = np.vstack((bw_0,bw_1));
			bw = np.vstack((bw,bw_2));

			delt = np.dot(inv(M), bw);

			dx = delt[0:m];

			ds = delt[(m):m2];

			dy = delt[(m2+1):sm];


			count = count + 1
			




