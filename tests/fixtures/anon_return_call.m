function g = anon_return_call(k)
    % A handle cannot be both returned and called in the closure-value subset.
    g = @(x) x + k;
    y = g(1);
end
