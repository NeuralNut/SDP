% Program: mat_s.m
% Description: This function is required by programs sdp.m and 
% sdp_pc.m (Algorithms 14.1 and 14.2, respectively). It coverts 
% a vector ofsize n(n+1)/2 to a symmetric matrix of size n x n 
% such that it is the inverse operator of svec defined by Eq.
% (14.37).
% Input:      
%    x: an input vector of size n(n+1)/2
% Output:
%    Z: a symmetric matrix of size n x n that is the inverse 
%       of operator svec in Eq. (14.37)
% =========================================================
function Z = mat_s(x)
x = x(:);
Nz = length(x);
s2i=1/sqrt(2);
n = round(0.5*(sqrt(8*Nz+1)-1));
Z = zeros(n);
cw = 0;
for i = 1:n-1,
  Z(i:n,i) = x(cw+1:cw+n-i+1);
  Z(i+1:n,i) = s2i*Z(i+1:n,i);
  Z(i,i:n) = Z(i:n,i)';
  cw = cw + n - i +1;
end
Z(n,n) = x(Nz);