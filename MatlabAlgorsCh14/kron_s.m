% Program: kron_s.m
% Description: This function is required by programs sdp.m and
% sdp_pc.m (Algorithms 14.1 and 14.2, respectively). It computes
% the symmetric Kronecker product of two n x n matrices M and N, 
% defined in Sec. 14.4.2, see Eq. (14.36).
% Input:      
%    M and N: input matrices
% Output:
%    Z: symmetric Kronecker product of matrices M and N
% Example:
% Evaluate the symmetric Kronecker product of matrices M and 
% N where M = [1 -1 2; -1 0 1; 2 1 4] and 
% N = [3 1 0; 1 2 -1; 0 -1 -2].
% Solution: 
% Execute the commands:
% M = [1 -1 2; -1 0 1; 2 1 4]
% N = [3 1 0; 1 2 -1; 0 -1 -2]
% P = kron_s(M,N)
% =====================================================
function Z = kron_s(M,N)
n=size(M)*[0 1]';
s2i=1/sqrt(2);
nz = n*(n+1)/2;
Z = zeros(nz);
nz = 0; 
for j = 1:n,
  for i = j:n,
    K = zeros(n,n);
    nz = nz + 1;
      if i == j,
        K(i,j) = 1;
        else
        K(i,j) = s2i;
        K(j,i) = s2i;
     end
    Z(:,nz) = svec(0.5*(N*K*M'+M*K*N'));
  end
end