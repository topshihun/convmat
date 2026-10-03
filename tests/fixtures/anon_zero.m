function y = anon_zero(a)
    f = @() a * 2;
    y = f();
end
