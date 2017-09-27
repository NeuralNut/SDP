% Program: sdp.m
% Title: SDP algorithm
% Description: Implements the primal-dual path-following 
% SDP algorithm (Algorithm 14.1).
% Theory: See Practical Optimization Secs. 14.4.1 - 14.4.4.
% Input:
%   {X0,y0,S0}: strictly feasible initial set
%           Ag: n x pn matrix that collects symmetric matriax
%               n x n matricies Ai for i = 1,...,p
%            b: [b1 ... bp]'in Eqs.(14.1b) and (14.4a)
%            C: n x n symmetric matrix in (14.1a) and (14.4b) 
%          gam: gamma in Eq. (14.45)
%         epsi: tolerance for duality gap
% Output:   
%    {X,y,S}: primal-dual solution up to the given tolerance
%          k: number of iterations at convergence
% Example: 
% Solve the problem in Example 14.1 by using Algorithm 14.1. 
% Solution: 
% Execute the following commands:
% X0 = eye(3)/3; y0 = [0.2 0.2 0.2 -4]'
% S0 = [2 0.3 0.4; 0.3 2 -0.6; 0.4 -0.6 1]
% A0 = [2 -0.5 -0.6; -0.5 2 0.4; -0.6 0.4 3]
% A1 = [0 1 0; 1 0 0; 0 0 0]
% A2 = [0 0 1; 0 0 0; 1 0 0]
% A3 = [0 0 0; 0 0 1; 0 1 0]
% A4 = eye(3)
% Ag = [A1 A2 A3 A4]
% b = [0 0 0 1]'
% C = -A0
% gam = 0.9 
% epsi = 1e-3
% [X,y,S,k] = sdp(X0,y0,S0,Ag,b,C,gam,epsi)
% ===============================================
function [X,y,S,k] = sdp(X0,y0,S0,Ag,b,C,gam,epsi)
disp(' ')
disp('Program sdp.m')
% Data preparation.
b = b(:);
p = length(b);
n = size(C)*[1 0]';
n2 = n*(n+1)/2;
X = X0;
y = y0(:);
S = S0;
I = eye(n);
gap = sum(sum(X.*S))/n;
sig = n/(15*sqrt(n)+n);
tau = sig*gap;
k = 0;
A = zeros(p,n2);
for i = 1:p,
   A(i,:) = (svec(Ag(:,(i-1)*n+1:i*n)))';
end
% SDP iterations. 
 while gap > epsi,
   % Solve Eq. (14.43) for {DX, dy, DS}.
   E = kron_s(S,I);
   F = kron_s(X,I);
   rc = svec(tau*I-0.5*(X*S+S*X));
   x = svec(X);
   rp = b-A*x;
   rd = svec(C-S-mat_s(A'*y));
   f1 = F*rd - rc;
   Ei = inv(E);
   M1 = A*Ei;
   M = M1*F*A';
   dy = inv(M)*(rp+A*Ei*f1);
   ad = A'*dy;
   dx = -Ei*(f1 - F*ad);
   ds = rd - ad;
   DX = mat_s(dx);
   DS = mat_s(ds);
   % Compute alpha and beta.
   Xi = inv(chol(X));
   Si = inv(chol(S));
   lx = min(eig(Xi'*DX*Xi));
   ls = min(eig(Si'*DS*Si));
     if lx >= 0,
      al = 1;
     else 
      al = min([1 -gam/lx]);
     end
     if ls >= 0,
      bl = 1;
     else
      bl = min([1 -gam/ls]);
     end
   % Update {X, y, S}
   X = X + al*DX;
   y = y + bl*dy;
   S = S + bl*DS;
   k = k + 1;
   % Evaluate duality gap and tau.
   gap = sum(sum(X.*S))/n;
   tau = sig*gap;
 end