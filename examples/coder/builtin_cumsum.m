function y = builtin_cumsum()
% `cumsum` / `cumprod` are not lowered yet.
y = cumsum([1, 2, 3, 4]);
end
