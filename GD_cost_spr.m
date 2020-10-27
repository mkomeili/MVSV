%M. Komeili, N. Armanfard, D. Hatzinakos, (2020), “Multiview Feature Selection for Single-view Classification”, IEEE Transactions on Pattern Analysis and Machine Intelligence.
%
function f=GD_cost_spr(alpha, P, Sm, v, grad, lambda,gama) 

n_m=size(Sm,2);
v = v-alpha*grad;
w = (v.^2);
p_m = Sm'*w; %margin
Logis = 1./(1+exp(-p_m)); 

index = Logis==0;
Logis(index) = 10^(-10);
f_m=(1/n_m)*sum(log(Logis));
f=gama*sum(w) + lambda*(w'*P*w) - f_m;
