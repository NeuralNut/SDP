% Program: socp_mt.m
% Title: Primal-dual interior-point algorithm for SOCP
% problems.
% Description: Implements the primal-dual path-following
% interior-point algorithm by Monteiro and Tsuchiya 
% (Algorithm 14.5). 
% Theory: See Practical Optimization Sec. 14.8.  
% Input: 
%   {A,b,c}: data set defined by Eq. (14.101) and in 
%   Sec. 14.8.1
%   {x0,s0,y0}: strictly feasible initial set
%   sig: parameter sigma in Eq. (14.125c)
%   n: n = [n1 n2 ... nq], see Eq. (14.101)
%   epsi: tolerance for duality gap
% Output:   
%    {x,s,y}: a primal-dual solution 
%    fs: objective function as solution point x
%    k: number of iterations at convergence
% Example: 
% Solve the problem in Example 14.5 by using Algorithm 14.5.
% Solution: 
% Execute the commands:
% [A,b,c,x0,s0,y0,n] = data_ex14_5
% sig = 1e-5
% epsi = 1e-4
% [x,s,y,fs,k] = socp_mt(A,b,c,x0,s0,y0,n,sig,epsi)
% ======================================================
function [x,s,y,fs,k] = socp_mt(A,b,c,x0,s0,y0,n,sig,epsi)
disp(' ')
disp('Program socp_mt.m')
% Data preparation
q = length(n);
m = sum(n);
m2 = 2*m;
nb = length(b);
Im = eye(m);
e = ones(m,1);
x = x0;
s = s0;
y = y0;
gap = x'*s;
mu = gap/q;
k = 0;
% SOCP iterations.
while mu >=  epsi,
   % Solve Eq. (14.125) for {dx,ds,dy}.
   [X,S] = get_xs(x,s,n);
   sm = 2*m + nb;
   M = zeros(sm,sm);
   M(1:nb,1:m) = A;
   M((nb+1):(nb+m),(m+1):m2) = Im;
   M((nb+1):(nb+m),(m2+1):sm) = A';
   M((nb+m+1):sm,1:m2) = [S X];
   bw = [b-A*x; c-s-A'*y; sig*mu*e-X*s];
   delt = inv(M)*bw;
   dx = delt(1:m);
   ds = delt((m+1):m2);
   dy = delt((m2+1):sm);
   % Perform line search using Eq. (14.126)
   a1 = find_alpha(x,dx,n);
   a2 = find_alpha(s,ds,n);
   t = c - A'*y;
   dt = -A'*dy;
   a3 = find_alpha(t,dt,n);
   a = 0.5*min([a1 a2 a3]);
   % Update {x,y,s}.
   x = x + a*dx;
   s = s + a*ds;
   y = y + a*dy;
   k = k + 1;
   % Evaluate duality gap.
   gap = x'*s;
   mu = gap/q;
 end
fs = c'*x;