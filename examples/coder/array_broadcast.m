function y = array_broadcast()
% Implicit expansion (array + row vector) is not lowered yet.
A = [1, 2; 3, 4];
b = [10, 20];
y = A + b;
end
