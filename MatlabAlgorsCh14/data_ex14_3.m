% Program: data_ex14_3.m
% Title: Data matrices of Example 14.3.
% Description: Loads the data matrices of Example 14.3 into the 
% MATLAB environment and generates matrices F0 and FF for
% functions projective_sdp and projective_feasi.
% Input: None.     
% Output:    
%   Matrices F0 and FF
% Example:
% Execute
%   [F0,FF] = data_ex14_3
% =================================================
function [F0,FF] = data_ex14_3
F0 = [0.50 0.55 0.33 2.38;
      0.55 0.18 -1.18 -0.40;
      0.33 -1.18 -0.94 1.46;
      2.38 -0.40 1.46 0.17];
F1 = [5.19 1.54 1.56 -2.80;
      1.54 2.20 0.39 -2.50;
      1.56 0.39 4.43  1.77;
      -2.80 -2.50 1.77 4.06];
F2 = [-1.11 0 -2.12 0.38;
       0 1.91 -0.25 -0.58;
      -2.12 -0.25 -1.49 1.45;
       0.38 -0.58 1.45 0.63];
F3 = [2.69 -2.24 -0.21 -0.74;
      -2.24 1.77 1.16 -2.01;
      -0.21 1.16 -1.82 -2.79;
      -0.74 -2.01 -2.79 -2.22];
F4 = [0.58 -2.19 1.69 1.28;
     -2.19 -0.05 -0.01 0.91;
      1.69 -0.01 2.56 2.14;
      1.28 0.91 2.14 -0.75];
FF = [F1 F2 F3 F4];