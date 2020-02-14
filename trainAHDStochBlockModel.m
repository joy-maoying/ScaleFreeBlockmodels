function [model,latnVar,varPar] = trainAHDStochBlockModel(N,M,Y,K)
% function [model,latnVar] = trainHDStochBlockModel(Y,K)
% training a Scale-free blockmodel by variational EM
% input: N, M, node and link numbers
%        Y, M by 2 matrix, each row stores a link (head, end)
%        K, cluster number
% ouput: model.pi, K by 1, cluster prior probability
%        model.B, K by K, link generating probability
%        model.lam, scalar, exp distribution parameter
%        latnVar.z, N by1, clustering index
%        latnVar.del, N by 1, degree decay arameter

%


%% initialization
% model parameters
% model.pi = ones(1,K)/K;
model.pi = rand(1,K) + 0.01;
model.pi = model.pi/sum(model.pi);
model.lam = .009; % (0.2 failed)
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
varPar.del = zeros(N,1);
%% variational EM interations
YY_1 = [Y;[Y(:,2) Y(:,1)]]; % duplated links
Loc_1 = sub2ind([N,N], YY_1(:,1), YY_1(:,2));
Loc_diag = sub2ind([N,N], 1:N, 1:N);
YY_0 = ones(N,N); YY_0(Loc_1)=0; YY_0(Loc_diag)=0;
YY_0 = find(YY_0==1);
[YY_00(:,1),YY_00(:,2)] = ind2sub([N,N],YY_0);
YY_0 = YY_00;clear YY_00   %YY_0 is the N by N link matrix
for iter = 1:maxIter
%     get_elbo(YY_1,YY_0,model,varPar)
    % E step
    tic
    for iterE = 1:10
        % updating phi
        for iN = randperm(N)
            nonlink_iN = (1:N)';
            link_iN = YY_1(YY_1(:,1)==iN,2);
            nonlink_iN([link_iN; iN]) = 0;
            nonlink_iN = nonlink_iN(nonlink_iN>0);
            
            term1 = (1 + varPar.del(iN) + varPar.del(link_iN,1))'...
                *(varPar.phi(link_iN,:)*log(model.B' + 1e-10));
            term2 = zeros(1,K);
            for iK = 1:K
                term2(1,iK) = 0;
                for jK = 1:K
                    temp = varPar.phi(nonlink_iN,jK)'*(model.B(iK,jK).^(1+varPar.del(iN)+varPar.del(nonlink_iN)));
                    term2(1,iK) = term2(1,iK) + temp;
                end
            end
            term = log(model.pi + 1e-10) + term1 - term2;
            %term = term-mean(term);
            term = term-max(term);
            term = exp(term);
            varPar.phi(iN,:) = term/sum(term);
%                         objdebug(iterE,iN) = get_elbo(YY_1,YY_0,model,varPar);
            
        end
        
    end

%     get_elbo(YY_1,YY_0,model,varPar)
    % updating delta
    for iN = randperm(N)
        nonlink_iN = (1:N)';
        link_iN = YY_1(YY_1(:,1)==iN,2);
        nonlink_iN([link_iN; iN]) = 0;
        nonlink_iN = nonlink_iN(nonlink_iN>0);
        del = linspace(0,2,10);
        ff = zeros(1,length(del));
        for id = 1:length(del)
            term1 = zeros(K,K);term2=term1;
            for iK = 1:K
                for jK = 1:K
                    term1(iK,jK) = del(id)*log(model.B(iK,jK) + 1e-10)*sum( varPar.phi(iN,iK)...
                        *varPar.phi(link_iN,jK));
                    term2(iK,jK) = sum( varPar.phi(iN,iK)...
                        *varPar.phi(nonlink_iN,jK).*( model.B(iK,jK).^(1+del(id) + varPar.del(nonlink_iN)))) ;
                end
            end
            ff(id) = sum(term1(:)) -sum(term2(:)) - model.lam*del(id);
        end
        [ee,oo] = max(ff);
        varPar.del(iN) = del(oo(1));
%         objdebug(iN) = get_elbo(YY_1,YY_0,model,varPar);

    end
%     get_elbo(YY_1,YY_0,model,varPar)
    
    
    % M step
    
    % update pi
    model.pi = sum(varPar.phi,1);
    model.pi = model.pi/sum(model.pi);
%     get_elbo(YY_1,YY_0,model,varPar)
    
    % update B
    for iK = 1:K
        for jK = iK:K
            NB = 100;
            bb = linspace(1e-3,1,NB);
            ff = zeros(1,NB);
            for ii = 1:NB
                b = bb(ii);
                term1 = log(b + 1e-10)*sum( (1 + varPar.del(YY_1(:,1))...
                    + varPar.del(YY_1(:,2))).* varPar.phi(YY_1(:,1),iK)...
                    .*varPar.phi(YY_1(:,2),jK) );
                term2 = sum( varPar.phi(YY_0(:,1),iK)...
                    .*varPar.phi(YY_0(:,2),jK).*( b.^(1+varPar.del(YY_0(:,1)) + varPar.del(YY_0(:,2))) ) );
                
                ff(ii) = term1 - term2;
            end
            [ee,oo] = max(ff);
            model.B(iK,jK) = bb(oo);
            model.B(jK,iK) = bb(oo);
        end
    end

%     get_elbo(YY_1,YY_0,model,varPar)
    
    % calcualting objective function
    obj(iter + 1) = get_elbo(YY_1,YY_0,model,varPar);
    tt= toc;
    fprintf('%dth EM iteration in %f seconds likelihood is %f \n',iter,tt, obj(iter));
    
    
    if abs(obj(iter+1)-obj(iter))/abs(obj(iter+1))<1e-5
        fprintf('converged in %d EM steps \n', iter);
        break;
    end
end
[conf_prd,latnVar.z] = max(varPar.phi,[],2);
latnVar.del = varPar.del;
model.loglik = obj(2:iter+1);

function elbo = get_elbo(YY_1,YY_0,model,varPar)
K = length(model.pi);
N = length(varPar.del);
term1 = zeros(K,K);
for iK = 1:K
    for jK = 1:K
        term1(iK,jK) = log(model.B(iK,jK) + 1e-10)*sum( (1 + varPar.del(YY_1(:,1))...
            + varPar.del(YY_1(:,2))).*varPar.phi(YY_1(:,1),iK).*...
            varPar.phi(YY_1(:,2),jK) ) ...
            - sum( varPar.phi(YY_0(:,1),iK).*...
            varPar.phi(YY_0(:,2),jK).* model.B(iK,jK).^(1+varPar.del(YY_0(:,1))...
            +varPar.del(YY_0(:,2))));
    end
end
term1 = sum(sum(term1))/2;
term2 = sum(log(model.pi + 1e-10).*sum(varPar.phi,1)) - sum(sum(log(varPar.phi + 1e-10).*varPar.phi));
elbo = term1 + term2 + N*log(model.lam + 1e-10) - model.lam*sum(varPar.del);


