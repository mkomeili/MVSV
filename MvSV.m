%M. Komeili, N. Armanfard, D. Hatzinakos, (2020), “Multiview Feature Selection for Single-view Classification”, IEEE Transactions on Pattern Analysis and Machine Intelligence.
%
%Inputs:
% X: input data in view x. Each column is a sample.
% Y: input data in view y. Each column is a sample.
% id: class label of the samples. 
% Yid: class label of te samples in Y
% sigma: it is hyper-parameter. It is kernel width of the Gaussian kernel. See the paper.
% lambda: it is a hyper-parameter that controls the cross-view matching error. See the paper.
% gama: it is a hyper-parameter that controls the L1 regularization term. See the paper.
%
%Output:
% w1: feature weight vector for view x. 
% w2: feature weight vector for view y.

function w1 = MvSV(X,Y,id, sigma,lambda,gama, X_tst,X_tst_id)

itr_max = 500;
f1=size(X,1);
f2=size(Y,1);
NRef=length(id)-1;
w=ones(f1+f2,1)./sqrt(f1+f2);
w1=w(1:f1);
w2=w(f1+1:f1+f2);

Difference =1;
t=0;
theta =[];
w_old = w;
while  (Difference>0.01) && (t<=4) 
    t=t+1;
    Z_x  = computeZ( X,id,w1,sigma);
    Z_y  = computeZ( Y,id,w2,sigma);
    Sm = [Z_x , zeros(f1,size(Z_y,2)) ; zeros(f2,size(Z_x,2)), Z_y ];
    
    lb=zeros(f1+f2,1);
    x0=w;
    
    P=zeros(f1+f2);
    for ss = 1:NRef
        ind_mN=1:NRef; ind_mN(ss)=[];
        Dxi = abs(bsxfun(@minus,X(:,ind_mN),X(:,ss)));
        Dyi = abs(bsxfun(@minus,Y(:,ind_mN),Y(:,ss)));
        P=P+[Dxi*Dxi' -Dxi*Dyi'; -Dyi*Dxi' Dyi*Dyi'];
    end
    P=P/NRef;
    
    [w, fval]  = GD_solver_spr( x0,P,Sm,lambda,gama,itr_max );
  
    w1=w(1:f1);
    w2=w(f1+1:f1+f2);
    
    Difference = norm(abs(w/max(w)-w_old/max(w_old)));
    theta(t) = Difference;
    w_old = w;
end


