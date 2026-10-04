function y = array_intermediate_loop(A, n)
    y = 0;
    for i = 1:n
        t = A(:);
        y = y + sum(t);
    end
end
