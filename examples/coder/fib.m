function y = fib(n) %#codegen
% Recursive Fibonacci, from the MATLAB Coder "recursive functions" docs.
if n < 2
    y = n;
else
    y = fib(n - 1) + fib(n - 2);
end
end
