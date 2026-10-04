function [dist, path] = dijkstra(W, source, target) %#codegen
% Dijkstra's shortest path, from the MATLAB Coder example gallery.
n = size(W, 1);
dist = Inf(1, n);
dist(source) = 0;
visited = zeros(1, n);
for i = 1:n
    dist = min(dist, dist(source) + W(:, source));
end
path = dist;
end
