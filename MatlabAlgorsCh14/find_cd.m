% Program: find_cd.m
% Description: This function is required by projective_sdp.m
% (Algorithm 14.4). It computes matrices Ck and Dk defined
% in Eqs.(14.91a) and (14.91b).
% Input:      
%   Xki: inverse of matrix Xk
%    ct: (p+1)-dimensinal vector in Eq.(4.2)
%    FF: an mz x mz*pz matrix of the form FF = [F1 F2 ... Fn]
%    F0: constant term in constraint F(x) >= 0
% Output:
%    Ck and Dk: matrices defined in Eq.(4.19)
%    xc and xd: coefficient vectors to express Ck and Dk as linear
%               combination of matrices F~i's (see Problem 14.15).
% =============================================================
function [Fx,Fwi,xc,xd] = find_cd(ct,Xki,FFt,mz,pz)
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
Fwi = inv(Fw);
xc = Fwi*ct;
xd = Fwi(:,end);