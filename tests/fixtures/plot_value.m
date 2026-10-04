function y = plot_value()
    % `plot` has no numeric result; using it as a value is deferred.
    x = [1, 2, 3];
    y = plot(x);
end
