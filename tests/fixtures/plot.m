function plot_demo()
    % Graphics have no C representation: `plot` lowers to a documented no-op.
    x = [1, 2, 3];
    plot(x);
end
