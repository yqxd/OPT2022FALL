function f=convergence_factor(values, sigma)
    dis = abs(values-sigma);
    sortdis = sort(dis);
    f= sortdis(1)/sortdis(2);
end