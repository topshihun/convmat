function y = array_concat()
% Horizontal concatenation of two row vectors: [1 2] and [3 4] make [1 2 3 4].
a = [1, 2];
b = [3, 4];
c = [a, b];
y = sum(c);
end
