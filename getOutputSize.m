function [ output_args ] = getOutputSize(n, p, s, f)
%GETOUTPUTSIZE Summary of this function goes here
%   Detailed explanation goes here

output_args = floor((n+2*p-f)/s + 1);
end

