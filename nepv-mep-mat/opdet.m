function [Deltai] = opdet(W,i)
    % Return Delta_i of the given multiparameter eigenvalue problem.
    % W should be of size m x (m+1).
    % The first cell column corresponds to the left hand side of the
    % equations, while the other coluns are considered as the right hand
    % side.
    
    WW = W(:,2:end);
    if i>0
        WW(:,i) = W(:,1);
    end
    
    n = size(WW{1,1},1);
    p = size(WW,1);
    sigmas = perms(1:p);
    N = size(sigmas,1);
    Deltai = sparse(n^p,n^p);

    for i = 1:N
        tsgn = sgn_perm(sigmas(i,:));
        tempdelt = tsgn*WW{1,sigmas(i,1)};
        for j = 2:p
            tempdelt = kron(tempdelt,WW{j,sigmas(i,j)});
        end
        Deltai = Deltai+tempdelt;
    end
end

