function y = builtin_predicates(x)
% `isnan` / `isinf` / `isempty` predicates are not lowered yet.
y = isnan(x) + isinf(x);
end
