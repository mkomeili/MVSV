function  [w, fval]  = GD_solver( x0,P,Sm,lambda,itr_max )
if nargin<5
    itr_max=500; 
end
n_m=size(Sm,2);
CostDiff = inf; Cost =10;
v=sqrt(x0);
j=1;
while (CostDiff>0.001*Cost(j)) && (j<itr_max) 
    j= j+1;
    p_m = Sm'*(v.^2); % Margin
    Logis = 1./(1+exp(-p_m)); % Probability of being true class. G(W'Z)
    Logis_der = 1 - Logis; 
    grad_m = 1/n_m*(Sm*Logis_der);
    grad=(lambda*2*P*(v.^2) - grad_m)*2.*v; 
    
    options = optimset('FunValCheck','off','TolX',1e-2);
    [alpha,Cost(j)] = fminbnd(@(alpha) GD_cost(alpha, P, Sm, v, grad, lambda),0, 1, options);
    v = v-alpha*grad;
    CostDiff = abs(Cost(j)-Cost(j-1));
end
w=v.^2;
if nargout > 1
    fval=Cost(j);
end