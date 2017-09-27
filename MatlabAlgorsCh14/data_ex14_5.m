% Program: data_ex14_5.m
% Title: Data matrices of Example 14.5.
% Description: Loads the data matrices/vectors of Example 14.5 into the 
% MATLAB environment for function socp_mt.
% Input: None.     
% Output:    
%   Matrices/vectors A,b,c,x0,s0,y0,n
% Example:
% Execute
%   [A,b,c,x0,s0,y0,n] = data_ex14_5
% =================================================
function [A,b,c,x0,s0,y0,n] = data_ex14_5
A = [ 1 0 0 0 0 0 0 0 0 0 0;
      0 -1 0 0 0 0 0.5 0 0 0 0;
      0 0 1 0 0 0 0 1 0 0 0;
      0 1 0 0 0 0 0 0 0 -0.7071 -0.3536;
      0 0 -1 0 0 0 0 0 0 -0.7071 0.3536];
b =[1 0 0 0 0]';
c = [0 0 0 0 0 1 -0.5 0 1 4.2426 -0.7071]';
x0 = [1 0 0 0 0 0.1 0 0 0.1 0 0]';
s0 = [3.7 1 -3.5 0 0 1 0.25 0.5 1 -0.35355 -0.1767]';
y0 = [-3.7 -1.5 -0.5 -2.5 -4]';
n = [5 3 3];