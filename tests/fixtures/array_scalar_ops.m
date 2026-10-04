function y = array_scalar_ops(A, k)
    B = A(:);
    y = (2 - B) ./ k + B .* k;
end
