% Program: sdp_pc.m
% Title: Predictor-corrector algorithm for SDP problems
% Description: Implements the primal-dual path-following
% algorithm (Algorithm 14.2). It incorporates Mehrotra's 
% predictor-corrector rule. 
% Theory: See Practical Optimization Sec. 14.5.  
% Input: 
%   {X0,y0,S0}: strictly feasible initial set
%           Ag: n x pn matrix that collects symmetric 
%               matrices Ai for i = 1,...,p.
%            b: b = [b1 ... bp]'in Eqs.(14.1b) and (14.4a)
%            C: n x n symmetric matrix in Eqs. (14.1a) and 
%               (14.4b) 
%          gam: gamma in Eqs. (14.53) and (14.58).
%         epsi: tolerance for duality gap.
% Output:   
%      {X,y,S}: primal-dual solution up to the given tolerance
%            k: number of iterations at convergence
% Example: 
% Solve the problem in Example 14.1 by using Algorithm 14.1. 
% Solution: 
% Execute the commands:
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
% ==================================================
function [X,y,S,k] = sdp_pc(X0,y0,S0,Ag,b,C,gam,epsi)
disp(' ')
disp('Program sdp_pc.m')
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
k = 0;
A = zeros(p,n2);
for i = 1:p,
   A(i,:) = (svec(Ag(:,(i-1)*n+1:i*n)))';
end
% SDP iterations. 
while gap > epsi,
  % Generate predictor direction.
   Xi = inv(chol(X));
   Si = inv(chol(S));
   E = kron_s(S,I);
   F = kron_s(X,I);
   x = svec(X);
   rp = b-A*x;
   rd = svec(C-S-mat_s(A'*y));
   Ei = inv(E);
   M1 = A*Ei;
   M = M1*F*A';
   Mi = inv(M);
   rc = svec(-0.5*(X*S+S*X));  
   f1 = F*rd - rc;
   dy = Mi*(rp+A*Ei*f1);
   ad = A'*dy;
   dx = -Ei*(f1 - F*ad);
   ds = rd - ad;
   DX = mat_s(dx);
   DS = mat_s(ds);   
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
   Xp = X + al*DX;
   yp = y + bl*dy;
   Sp = S + bl*DS;
   sig = (sum(sum(Xp.*Sp))/sum(sum(X.*S)))^3;
   gap = sum(sum(X.*S))/n;
   tau = sig*gap;
   % Generate corrector direction.
   rc = svec(tau*I-0.5*(X*S+S*X+DX*DS+DS*DX));
   f1 = F*rd - rc;
   dy = Mi*(rp+A*Ei*f1);
   ad = A'*dy;
   dx = -Ei*(f1 - F*ad);
   ds = rd - ad;
   DX = mat_s(dx);
   DS = mat_s(ds);
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
   % Update {X, y, S}.
   X = X + al*DX;
   y = y + bl*dy;
   S = S + bl*DS;
   % Evaluate duality gap.
   gap = sum(sum(X.*S))/n;
   k = k + 1; 
end