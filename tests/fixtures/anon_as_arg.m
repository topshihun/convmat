function y = anon_as_arg(a)
    f = @(x) x + a;
    y = sin(f);
end
