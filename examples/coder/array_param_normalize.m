function y = array_param_normalize(A)
% Center a dynamic-shape vector by subtracting its own mean (runtime matrix).
m = mean(A);
y = A(:) - m;
end
