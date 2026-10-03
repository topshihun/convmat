function y = anon_struct_field(a)
    s.f = @(x) x + a;
    y = s.f(1);
end
