function y = zz_arr()
    A = [1, 2, 3];
    f = @(x) x .* x;
    y = f(A);
end
