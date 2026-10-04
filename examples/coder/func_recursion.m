function y = func_recursion(n)
% Recursive function calls are not lowered yet.
if n <= 1
    y = 1;
else
    y = n * func_recursion(n - 1);
end
end
