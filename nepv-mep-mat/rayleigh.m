function [lambda,res] = rayleigh(W,X,f)
    D1X = -W{2,1}*X*W{1,3}.'+W{2,3}*X*W{1,1}.';
    D0X = W{2,3}*X*W{1,2}.'-W{1,2}*X*W{1,3}.';
    lambda = (X(:)'*D1X(:)) / (X(:)'*D0X(:));
    x = X(:,1)/norm(X(:,1));
    res = norm((-W{1,1} + lambda*W{1,2} + f(x)*W{1,3})*x);
end