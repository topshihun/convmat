function f = func_handle_return(k)
% Returning an anonymous-function handle needs the dynamic closure tier.
f = @(x) x + k;
end
