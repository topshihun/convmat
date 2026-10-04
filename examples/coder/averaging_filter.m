function avg = averaging_filter(x) %#codegen
% MATLAB Coder tutorial "Generate C Code from MATLAB Code".
% 16-sample moving average held in a persistent buffer.
persistent buffer;
if isempty(buffer)
    buffer = zeros(16, 1);
end
buffer = [x; buffer(1:end-1)];
avg = mean(buffer);
end
