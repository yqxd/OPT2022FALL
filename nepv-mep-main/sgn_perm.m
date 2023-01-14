function s = sgn_perm(permVec)
    % Returns the sign (+1 or -1) of the given permutation.
    
    n = length(permVec);
    permMat = zeros(n,n);
    for i = 1:n
        permMat(i, permVec(i)) = 1;
    end
    s=det(permMat);
end