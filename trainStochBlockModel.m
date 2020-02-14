function [model,latnVar,varPar] = trainStochBlockModel(N,M,Y,K)
% function [model,latnVar] = trainSFBlockModel(Y,K)
% training a Scale-free blockmodel by variational EM
% input: N, M, node and link numbers
%        Y, M by 2 matrix, each row stores a link (head, end)
%        K, cluster number
% ouput: model.pi, K by 1, cluster prior probability
%        model.B, K by K, link generating probability
%        latnVar.z, N by1, clustering index
%        varPar, variational parameter 





%% initialization
% model parameters
model.pi = rand(1,K) + 0.01;
model.pi = model.pi/sum(model.pi);
probAver = M/(0.5*N*(N-1));
model.B = ones(K,K)*probAver;
model.B(sub2ind([K,K], 1:K, 1:K)) = min(0.7, 10*probAver);

% training parameter
maxIter = 100;
obj = zeros(1,maxIter);
obj(1) = -inf;


% variational parameters
varPar.phi = rand(N,K);
varPar.phi = bsxfun(@times,varPar.phi,1./sum(varPar.phi,2));

%% variational EM interations
YY_1 = [Y;[Y(:,2) Y(:,1)]]; % duplated links
Loc_1 = sub2ind([N,N], YY_1(:,1), YY_1(:,2));
Loc_diag = sub2ind([N,N], 1:N, 1:N);
YY_0 = ones(N,N); YY_0(Loc_1)=0; YY_0(Loc_diag)=0;
YY_0 = find(YY_0==1);
[YY_00(:,1),YY_00(:,2)] = ind2sub([N,N],YY_0);
YY_0 = YY_00;  clear YY_00  % YY_0  network matrix
for iter = 1:maxIter
    % E step
    tic
    for iterE = 1:10
        % updating phi
        for iN = randperm(N)  % phi_i 's i
            nonlink_iN = (1:N)';
            link_iN = YY_1(YY_1(:,1)==iN,2); % j
            nonlink_iN([link_iN; iN]) = 0;
            nonlink_iN = nonlink_iN(nonlink_iN>0);
            
            term1 = sum(varPar.phi(link_iN,:)*log(model.B' + 1e-10),1);
            term2 = sum(varPar.phi(nonlink_iN,:)*model.B',1);
            term = log(model.pi + 1e-10) + term1 - term2;
            %term = term - mean(term);
            term = term - max(term);
            term = exp(term);
            if isnan(term(1)) 
                a;
            end
            varPar.phi(iN,:) = term/sum(term);
        end
                
    end
        
   % M step
   
% update pi
   model.pi = sum(varPar.phi,1);
   model.pi = model.pi/sum(model.pi);
   
   % update B

   for iK = 1:K
       for jK = iK:K
           NB = 10;
           bb = linspace(1e-3,1,NB);
           ff = zeros(1,NB);
           for ii = 1:NB
               b = bb(ii);
               term1 = log(b+1e-10)*sum(varPar.phi(YY_1(:,1),iK)...
                   .*varPar.phi(YY_1(:,2),jK));
               term2 = sum( varPar.phi(YY_0(:,1),iK)...
                   .*varPar.phi(YY_0(:,2),jK)...
                   *b );
               ff(ii) = term1 - term2;
               
           end
           [ee,oo] = max(ff);
           
           model.B(iK,jK) = bb(oo);
           model.B(jK,iK) = bb(oo);
       end
   end
   
   % calcualting objective function
   obj(iter + 1) = get_elbo(YY_1,YY_0,model,varPar);
   tt= toc;
   fprintf('%dth EM iteration in %f seconds, the linklihood is %f \n',iter,tt, obj(iter));
   
    
   if abs(obj(iter+1)-obj(iter))/abs(obj(iter+1))<1e-5
       fprintf('converged in %d EM steps \n', iter);
       break;
   end
end

[conf_prd,latnVar.z] = max(varPar.phi,[],2);
model.loglik = obj(2:iter+1);

function elbo = get_elbo(YY_1,YY_0,model,varPar)
K = length(model.pi);
term1 = zeros(K,K);
for iK = 1:K
    for jK = 1:K
        term1(iK,jK) = log(model.B(iK,jK) + 1e-10)*sum( varPar.phi(YY_1(:,1),iK).*...
            varPar.phi(YY_1(:,2),jK) ) ...
            - sum( varPar.phi(YY_0(:,1),iK).*...
            varPar.phi(YY_0(:,2),jK))*model.B(iK,jK);
    end
end
term1 = sum(sum(term1))/2;
term2 = sum(log(model.pi + 1e-10).*sum(varPar.phi,1)) - sum(sum(log(varPar.phi+1e-10).*varPar.phi));
   elbo = term1 + term2;