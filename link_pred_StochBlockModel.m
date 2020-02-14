function roc = link_pred_StochBlockModel(model,latnVar,N,E,E_pred,opt)
% function roc = link_pred_StochBlockModel(model,latnVar,E,E_pred)
% link prediction by SBM
% input: model, model parameters
%        latnVar.z, node cluster index
%        latnVar.del, degree decay parameter
%        E, existing edges
%        E_pred, edges to be predicted
%        opt, if or not hetero-degree
% output: roc.fp, roc.tp 
K = size(model.B,1);
E_all = [E;E_pred];

%%% geting non-edges and their prob
Y = ones(N,N); Y = Y - eye(N);
Y = triu(Y);
Y(sub2ind([N,N],E_all(:,1),E_all(:,2))) = 0;
[E_non(:,1), E_non(:,2)] = ind2sub([N,N],find(Y==1));

M_non = size(E_non,1);

head_non_z = latnVar.z(E_non(:,1));
end_non_z = latnVar.z(E_non(:,2));
b_non_zz = model.B(sub2ind([K,K],head_non_z,end_non_z));

if nargin == 5
    prob_non = b_non_zz;
elseif strcmp(opt,'hd')
    del_non_zz = latnVar.del(head_non_z) + latnVar.del(end_non_z);
    prob_non = b_non_zz.^(1+del_non_zz);
end

%%%% getting pred-edges prob
M_pred = size(E_pred,1);


head_pred_z = latnVar.z(E_pred(:,1));
end_pred_z = latnVar.z(E_pred(:,2));
b_pred_zz = model.B(sub2ind([K,K],head_pred_z,end_pred_z));

if nargin==5
    prob_pred = b_pred_zz;
elseif opt == 'hd'
    del_pred_zz = latnVar.del(head_pred_z) + latnVar.del(end_pred_z);
    prob_pred = b_pred_zz.^(1+del_pred_zz);
end


%%% calculating roc

prob_thresh = linspace(0,1,100);
for ip = 1:100
    prob_tmp = prob_thresh(ip);
    roc.fp(ip) = sum(prob_non>prob_tmp)/M_non;
    roc.tp(ip) = sum(prob_pred>prob_tmp)/M_pred;
end
return