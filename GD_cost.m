%M. Komeili, N. Armanfard, D. Hatzinakos, (2020), “Multiview Feature Selection for Single-view Classification”, IEEE Transactions on Pattern Analysis and Machine Intelligence.
%

function f=GD_cost(alpha, P, Sm, v, grad, lambda) 

n_m=size(Sm,2);
v = v-alpha*grad;
p_m = Sm'*(v.^2); 
Logis = 1./(1+exp(-p_m)); 

index = Logis==0;
Logis(index) = 10^(-10);

f_m=(1/n_m)*sum(log(Logis));

f=lambda*(v.^2)'*P*(v.^2) - f_m;
