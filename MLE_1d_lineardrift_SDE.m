% parameters setup for  dx=-a*x^3*dt-g*x*dt+s*dw1+sigma*x*dw2
a=1;
g=1;
s=1;
sigma=1;
xzero=0;     %initial value for x
dt=0.01;     %step length for discretization
N_dt=10^6;   % number of discretization
N_sample=1;  %number of samples
N_theta=4;   %size of parameters collection
theta_true=[a;g;s;sigma];
theta_sample=zeros(N_theta,N_sample);  %space for storage MLE of samples
%MLE for N_sample samples
for m=1:N_sample
   x=generating_SDE(a,g,s,sigma,xzero,N_dt,dt);  
   theta_guess=initial_guess(x);
   options = optimoptions('fminunc','Display','iter-detailed','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective');
   [theta_sample(:,m),MLE] = fminunc(@negative_log_likelihood,theta_guess.',options);
%     sens_plot(x,theta_guess,theta_sample(:,m));
end
%  [sample_mean, sample_rel_err]= sample_statistics(theta_sample_1_2,theta_true);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x=generating_SDE(a,g,s,sigma,xzero,N,dt)
%Milstein method: approximate numerical solution: dx=-ax^3 dt - gx dt+sqrt(s^2+sigma^2x^2)dw
x=zeros(N+1,1);
x(1)=xzero;
for i=1:N
    dw1=sqrt(dt)*randn;
    dw2=sqrt(dt)*randn;
    x(i+1)=x(i) -a*x(i)^3*dt - g*x(i)*dt + s*dw1 + sigma*x(i)*dw2 + 0.5*sigma^2*x(i)*(dw2^2-dt);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function theta_guess= initial_guess(x) 
dt= evalin('base','dt');
N = evalin('base','N_dt');
    theta_guess=zeros(4,1);
    A=zeros(2,2);
    B=zeros(2,1);
    for i=1:N
      A(1,1)=A(1,1)+x(i)^6;
      A(1,2)=A(1,2)+x(i)^4;
      A(2,2)=A(2,2)+x(i)^2;
      B(1,1)=B(1,1)- (x(i+1)-x(i))*x(i)^3;
      B(2,1)=B(2,1)- (x(i+1)-x(i))*x(i);
    end
    A(2,1)=A(1,2);
    A=A*dt;
    theta_guess([1 2],1)=inv(A)*B;
    p1=0;
    for i=1:N
      p1 = p1 + (x(i+1) - x(i) + theta_guess(1)*x(i)^3*dt + theta_guess(2)*x(i)*dt)^2;
    end
    theta_guess(3)= sqrt(p1/(N*dt));
    theta_guess(4)= sqrt((p1-theta_guess(3)^2*dt)/A(2,2));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f,G,H] = negative_log_likelihood(theta)
dt= evalin('base','dt');
N = evalin('base','N_dt');
x = evalin('base','x');
f=0;  % negative log-likelihood function, which we want to minimize
G=zeros(4,1);  %gradient 
H=zeros(4,4);  %hessian
    for i=1:N
        p = x(i+1) -x(i)+theta(1)*x(i)^3*dt +theta(2)*x(i)*dt;
        q = theta(3)^2 +theta(4)^2 *x(i)^2;
        
        f=f + p^2 /(2*q*dt) + 0.5*log(2*pi*q*dt);
        
        G(1)=G(1) +x(i)^3 *p / q;
        G(2)=G(2) +x(i)* p / q;
        G(3)=G(3) -theta(3)*p^2 / (q^2*dt) + theta(3) / q;
        G(4)=G(4) -theta(4)*x(i)^2*p^2 / (q^2*dt) + theta(4)*x(i)^2 / q;
        
        H(1,1)=H(1,1) + x(i)^6*dt / q;
        H(1,2)=H(1,2) + x(i)^4*dt / q;
        
        H(1,3)=H(1,3) - 2*theta(3)*x(i)^3*p / (q^2);
        H(1,4)=H(1,4) - 2*theta(4)*x(i)^5*p / (q^2);
        H(2,2)=H(2,2) + x(i)^2*dt / q;
        H(2,3)=H(2,3) - 2*theta(3)*x(i)*p / (q^2);
        H(2,4)=H(2,4) - 2*theta(4)*x(i)^3*p / (q^2);
        
        H(3,3)=H(3,3) + (3*theta(3)^2 -theta(4)^2*x(i)^2)*p^2 / (q^3*dt) +(-theta(3)^2+theta(4)^2*x(i)^2) / (q^2);
        H(3,4)=H(3,4) + 4*theta(3)*theta(4)*x(i)^2*p^2 / (q^3*dt) -2*theta(3)*theta(4)*x(i)^2 / (q^2);
        H(4,4)=H(4,4) - x(i)^2*(theta(3)^2 -3*theta(4)^2*x(i)^2)*p^2 / (q^3*dt) + x(i)^2*(theta(3)^2-theta(4)^2*x(i)^2) / (q^2);   
    end
        H(2,1)=H(1,2);
        H(3,1)=H(1,3);
        H(4,1)=H(1,4);
        H(3,2)=H(2,3);
        H(4,2)=H(2,4);
        H(4,3)=H(3,4);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sample_mean, sample_rel_err]= sample_statistics(theta_sample,theta_true)
sample_mean=mean(theta_sample,2);
sample_rel_err=abs(sample_mean-theta_true)./theta_true;
fprintf('sample mean is %d\n',sample_mean);
fprintf('sample relative error os %d\n',sample_rel_err);
end

%sensitive plot for s and sigmain the log-likelihood function, fixed a=g=1
function sens_plot(x,theta_guess, theta_opt)
dt= evalin('base','dt');
N_dt = evalin('base','N_dt');
n_s=41;
n_sigma=41;
ss=linspace(0,4,n_s);
ssigma=linspace(0,4,n_sigma);
[X,Y] = meshgrid(ss,ssigma);

L_sen=zeros(n_s,n_sigma);
L_guess=0;
L_true=0;
for i=1:N_dt    
        p = x(i+1) -x(i)+ theta_opt(1)*x(i)^3*dt +theta_opt(2)*x(i)*dt; 
        q2 = theta_guess(3)^2 + theta_guess(4)^2*x(i)^2;
        L_guess = L_guess + p^2 /(2*q2*dt)+ 0.5*log(2*pi*q2*dt);
      
        q3 = theta_opt(3)^2 + theta_opt(4)^2*x(i)^2;
        L_true = L_true + p^2 /(2*q3*dt)+ 0.5*log(2*pi*q3*dt);
end

for k=1:n_s
    for j=1:n_sigma
      for i=1:N_dt
        p = x(i+1) -x(i)+ theta_opt(1)*x(i)^3*dt +theta_opt(2)*x(i)*dt;       
        q = ss(k)^2 +ssigma(j)^2 *x(i)^2;
        L_sen(j,k)=L_sen(j,k) +p^2 /(2*q*dt)+ 0.5*log(2*pi*q*dt);
      end
    end
end   

mesh(X,Y,L_sen);%3D-plot
hold on;
plot3(abs(theta_guess(3)),abs(theta_guess(4)),L_guess,'g*','MarkerSize', 15);
hold on;
plot3(abs(theta_opt(3)),abs(theta_opt(4)),L_true,'r*','MarkerSize', 15);
end
