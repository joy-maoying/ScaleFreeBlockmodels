function [E, E_dele] = dele_edge(E,N,m)
%function E = dele_edge(E,N), deleting edges from a graph
% input: N, node number
%        E, edge set, M by 2 matrix
%        m, number of edges to be deleted
% ouput: E, resulting edge set

m_dele = 0;

Y = sparse(E(:,1),E(:,2),1,N,N);
deg = sum(Y+Y',2);

idx_dele = zeros(m*2,1);
M = size(E,1); % edge number
while m_dele<m
    
    idx_dele_tmp = randsample(M,m);
    
    E_dele_tmp = E(idx_dele_tmp,:);
    
    for ii = 1:m
        idx_n = E_dele_tmp(ii,:);
        if any(deg(idx_n)<2) || any(idx_dele==idx_dele_tmp(ii))
            continue;
        else
            m_dele = m_dele + 1;
            idx_dele(m_dele) = idx_dele_tmp(ii);
            deg(idx_n) = deg(idx_n) - 1;
        end
    end
    
end
idx_dele = idx_dele(1:m);
E_dele = E(idx_dele,:);
E(idx_dele,:) = [];
