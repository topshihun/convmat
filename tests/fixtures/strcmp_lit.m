function y = strcmp_lit()
% strcmp of char literals folds to a logical scalar.
y = strcmp('abc', 'abd') + strcmp('x', 'x');
end
