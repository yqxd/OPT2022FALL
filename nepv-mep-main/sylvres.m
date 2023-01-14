function res= sylvres(W,sigma,X)
    X = X/norm(X(:,1));
    f = (-W{2,1}+sigma*W{1,2})*X*W{1,3}.'  - W{2,3}*X*(-W{1,1}+sigma*W{1,2}).'  -  W{2,3}*X*W{1,2}.' + W{1,2}*X*W{1,3}.';
    res = norm(f(:));
end