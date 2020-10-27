%M. Komeili, N. Armanfard, D. Hatzinakos, (2020), “Multiview Feature Selection for Single-view Classification”, IEEE Transactions on Pattern Analysis and Machine Intelligence.

function  Z_x  = computeZ( X,id,weight,sigma_margin)
f1=size(X,1);
NsRef=size(X,2);
Z_x = zeros(f1, NsRef);
for ii = 1:NsRef 
    x_ii = X(:,ii);
    subj = id(ii);
    ind_P = id==subj;
    ind_N = ~ind_P;
    Temp = abs(bsxfun(@minus,X,x_ii));
    dist = weight(:)'*Temp(:,ind_N);
    prob = exp(-dist/sigma_margin); %big sigma > all elements of prob are the same. sigma 0.01 > delta
    if sum(prob)~=0
        prob_1 = prob/sum(prob);
    else
        [~,I] = sort(dist); %I(1) is smallest
        prob(I(1))=1;
        prob_1=prob;
    end
    NM = Temp(:,ind_N)*prob_1(:);
    ind_P(ii)=0;
    dist = weight(:)'*Temp(:,ind_P);
    prob = exp(-dist/sigma_margin); %big sigma > all elements of prob are the same. sigma 0.01 > delta
    if sum(prob)~=0
        prob_1 = prob/sum(prob);
    else
        [~,I] = sort(dist); %I(1) is smallest
        prob(I(1))=1;
        prob_1=prob;
    end
    NH = Temp(:,ind_P)*prob_1(:);
    Z_x(:,ii)=NM-NH;
end
