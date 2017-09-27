% Program: svec.m
% Description: This function is required by programs sdp.m and 
% sdp_pc.m (Algorithms 14.1 and 14.2). It computes a column
% vector of dimension n(n+1)/2 froma symmetric matrix 
% of size n x n using Eq.(14.37).
% Input:      
%    X: input symmetric matrix
% Output:
%    z: vector of dimention n(n+1)/2 defined in Eq. (14.37)
% =========================================================
function z = svec(X)
n = size(X)*[1 0]';
s2 = sqrt(2);
z = [];
for i = 1:n-1,
 z = [z; X(i,i); s2*X(i+1:n,i)];
end
z = [z; X(n,n)];
z = z(:);