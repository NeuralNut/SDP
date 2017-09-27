% Program: find_alpha.m
% Description: This function is required by socp_mt.m
% (Algorithm 14.5). It computes alpha_k by using 
% Eq. (14.126).
% Input:      
%    x: current point xk (or sk)
%   dx: current increment dxk (or dsk)
%    n: n = [n1 n2 ... nq], see Eq. (14.101)
% Output:
%    a: value of alpha obtained
% =============================
function a = find_alpha(x,dx,n)
q = length(n);
aw = zeros(q,1);
act = 0;
for i = 1:q,
    ni = n(i);
    xi = x((act+1):(act+ni));
    dxi = dx((act+1):(act+ni));
    x1 = xi(1);
    xr = xi(2:ni);
    d1 = dxi(1);
    dr = dxi(2:ni);
    p0 = d1^2 - (norm(dr))^2;
    p1 = 2*(x1*d1 - xr'*dr);
    p2 = x1^2 - (norm(xr))^2;
    if d1 >= 0,
       aw1 = 1;
    else 
       aw1 = 0.99*(x1/(-d1));
    end
    rt = sort(roots([p0 p1 p2]));
    if p0 > 0 & rt(1) > 0,
       aw2 = rt(1);
    elseif p0 > 0 & rt(2) < 0,
       aw2 = 1;
    elseif p0 < 0,
       aw2 = rt(2);
    end
    aw(i) = min([aw1 aw2]);
    act = act + ni;
end
a = min(aw);