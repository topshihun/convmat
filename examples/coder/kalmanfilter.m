function y = kalmanfilter(z) %#codegen
% Scalar Kalman filter, from the MathWorks MATLAB Coder example gallery.
persistent x P
if isempty(x)
    x = 0;
    P = 1;
end
Q = 0.01;
R = 0.1;
P = P + Q;
K = P / (P + R);
x = x + K * (z - x);
P = (1 - K) * P;
y = x;
end
