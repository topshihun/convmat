function y = array_loop_sum(A)
    y = 0;
    for i = 1:numel(A)
        y = y + A(i);
    end
end
