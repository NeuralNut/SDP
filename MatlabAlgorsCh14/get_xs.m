% Program: get_xs.m
% Description: This function is required by socp_mt.m
% (Algorithm 14.5). It constructs matrices X and S 
% defined in Eqs.(14.125d) and (14.125e).
% Input:      
%    x: current point xk
%    s: current point sk
%    n: %   n: n = [n1 n2 ... nq], see Eq. (14.101)
% Output:
%    X: matrix X defined by Eqs.(14.125d)
%    S: matrix S defined by Eqs.(14.125e)
% ============================
function [X,S] = get_xs(x,s,n)
m = sum(n);
q = length(n);
X = zeros(m,m);
S = X;
ct = 0;
for i = 1:q,
    ni = n(i);
    xi = x((ct+1):(ct+ni));
    Ii = eye(ni-1);
    Xi = [xi'; xi(2:ni) xi(1)*Ii];
    X((ct+1):(ct+ni),(ct+1):(ct+ni)) = Xi;
    si = s((ct+1):(ct+ni));
    Si = [si'; si(2:ni) si(1)*Ii];
    S((ct+1):(ct+ni),(ct+1):(ct+ni)) = Si;
    ct = ct + ni;
end