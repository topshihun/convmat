function y = value_special()
% `Inf` / `NaN` literals, observed through the isinf/isnan predicates.
y = isinf(Inf) + isnan(NaN);
end
