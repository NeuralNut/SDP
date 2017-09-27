% Program: proj.m
% Description: This function is required by programs 
% projective_feasi.m (Algorithms 14.3) and projective_sdp  
% (Algorithm 14.4). It performs an orthogonal projection of the  
% input matrix Xk > 0 onto the subspace E = range(F)in the  
% metric <.,.> with respect to inv(Xk).
% Theory: See Practical Optimization Secs. 14.6.2 and 14.6.3.
% Input:
%    Xk: a positive matrix of size mz x mz
%   FFt: an (m+1) x (m+1)*(p+1) constraint matrix 
%        for argumented decision variables [x; tau]
% Output: 
%     x: a vectior of dimension nz such that 
%        Xkp = x(1)*F1 + x(2)*F2 + ... + x(n+1)*Fn+1   
%        minimizes the norm || Y - Xk || with respect to inv(Xk)
%   Xkp: Xkp = x(1)*F1 + x(2)*F2 + ... + x(n+1)*Fn+1
% =============================================================
function [x,Xkp] = proj(Xki,FFt,mz,pz)
Fx = Xki*FFt;
Fw = zeros(pz,pz);
for i = 1:pz,
   for j = 1:pz;
      if i > j,
         Fw(i,j) = Fw(j,i);
      else
         Fw(i,j) = trace(Fx(:,(i-1)*mz+1:i*mz)*Fx(:,(j-1)*mz+1:j*mz));
      end
   end
end
qw = zeros(pz,1);
for i = 1:pz,
   qw(i) = trace(Fx(:,(i-1)*mz+1:i*mz));
end
x = inv(Fw)*qw;
Xkp = zeros(mz,mz);
for i = 1:pz,
   Xkp = Xkp + x(i)*FFt(:,(i-1)*mz+1:i*mz);
end