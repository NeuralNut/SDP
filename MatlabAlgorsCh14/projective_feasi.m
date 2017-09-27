% Program: projective_feasi.m
% Title: Projective algorithm for homogenized strict-
%        feasibility problems.
% Description: Implements the projective algorithm (Algorithm
%              14.3). It finds a vector x = [x1 x2 ... xn]'
%              such that F(x) = x1*F1 + x2*F2 +... + xn*Fn 
%              + F0 is positive definite, where all Fi 
%             are symmetric matrices.
% Theory: See Practical Optimization Sec. 14.6.2. 
% Input:  
%    FF: an m x m*n matrix of the form FF = [F1 F2 ... Fn]
%    F0: constant term in constraint matrix F(x) > 0
% Output: 
%    xs: decision vector at which F(x) is positive definite
%    Xs: F(x) at x = xs
%     k: number of iterations at convergence
% Example: 
% Solve the problem in Example 14.3 by using Algorithm 14.3. 
% Solution: 
% Execute the commands:
% [F0,FF] = data_ex14_3
% [xs,Xs,k] = projective_feasi(FF,F0)
% =========================================
function [xs,Xs,k] = projective_feasi(FF,F0)
disp(' ')
disp('Program projective_feasi.m')
[m,nn] = size(FF);
n = round(nn/m);
nz = n + 1;
mz = m + 1;
FFe = [FF F0];
FFt = zeros(m+1,nz*(m+1));
FFt(m+1,end) = 1;
for i = 1:nz,
    FFt(1:m,(i-1)*(m+1)+1:i*(m+1)-1) = FFe(:,(i-1)*m+1:i*m);
end
% Data initialization.
I = eye(mz);
k = 0;
Xki = I;
[x,Xkp] = proj(Xki,FFt,mz,nz);
vi = min(eig(Xkp));
% Iteration begins.
while vi <= 0,
   Xw = Xki*Xkp - I;
   rou = max(abs(eig(Xw)));
   gk = 1/(1+rou);
   Xki = Xki - gk*Xw*Xki;
   [x,Xkp] = proj(Xki,FFt,mz,nz);
   vi = min(eig(Xkp));
   k = k + 1;
end
% Output results.
xs = x(1:n)/x(n+1);
Xs = F0;
for i = 1:n,
   Xs = Xs + xs(i)*FF(:,(i-1)*m+1:i*m);
end