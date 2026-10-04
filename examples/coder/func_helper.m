function y = func_helper(x)
% Calling another function in the same file is not lowered yet.
y = square(x) + 1;
end

function s = square(x)
s = x * x;
end
