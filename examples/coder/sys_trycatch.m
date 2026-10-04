function y = sys_trycatch(x)
% try/catch is not lowered yet.
try
    y = 1 / x;
catch
    y = 0;
end
end
