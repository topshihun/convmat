function f = func_handle_builtin()
% A handle to a built-in (`@sin`) is not in the scalar-handle subset.
f = @sin;
end
