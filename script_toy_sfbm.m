% a toy example for SF Blockmodel
clear;
close all
clc
% K = 3;
% mu = [1 0; 0 1; -1 0];
% 
% N = 30;
% z = repmat([1 2 3], 10, 1);
% z = z(:);
% var = 0.2;
% p1 = 0.2;
% p0 = 0.7;
% lam = log(p0)*(-0.4);
% muexp = 1/lam;
% for iN = 1:N
%     X(iN,:) = mu(z(iN),:) + randn(1,2)*var;
%     del(iN) = exprnd(muexp);
% end
% 
% B = ones(K,K)*p1;
% B = B + eye(K)*p0;
% 
% Link = zeros(N,N);
% for iN = 1:N-1
%     for jN = iN + 1:N
%         zi = z(iN);
%         zj = z(jN);
%         Link(iN,jN) = binornd(1,B(zi,zj)^(1+del(iN) + del(jN)));
%     end
% end
% 
% Link(16,17)=1;
% Link(21,22)=1;
% Link(27,28)=1;
% deg = sum(Link + Link',2);
% figure;hist(deg)
% [Y(:,1),Y(:,2)] = ind2sub([N,N],find(Link==1));
% 
% 
% save toy_data_sfbm


load toy_data_sfbm
figure;gscatter(X(:,1),X(:,2),z);axis equal
arrayfun(@(p,q)line([X(p,1),X(q,1)],[X(p,2),X(q,2)]),Y(:,1),Y(:,2));
M = size(Y,1);
[model,latnVar] = trainSFBlockModel(N,M,Y,K);

