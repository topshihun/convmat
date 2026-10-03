function y = anon_array_capture()
    A = [1, 2, 3];
    f = @(x) x + numel(A);
    y = f(1);
end
