%dx=(-a*x^3 -g*x)dt+s*dw1+sigma*x* dw2
% where a,g,s>0;x(0)=xzero;

a=1;
g=1;
s=1;
sigma=2;
xzero=0;
dt=0.01;
N=10^6;
M=100;
theta_sample=zeros(4,M);
for i=1:M
   x=generating_SDE(a,g,s,sigma,xzero,N,dt);
   [theta(i,:),MLE]=fmincon(@negative_log_likelihood,theta_guess.',[],[])
end;

%sensitive plot for s and sigmain the log-likelihood function, fixed a=g=1


n_s=31;
n_sigma=31;
ss=linspace(-3,3,n_s);
ssigma=linspace(-3,3,n_sigma);
L_sen=zeros(n_s,n_sigma);
for k=1:n_s
    for j=1:n_sigma
      for i=1:N
      p1 = x(i+1) -x(i)+a*x(i)^3*dt + g*x(i)*dt;
      p2 = ss(k)^2 +ssigma(j)^2 *x(i)^2;
      L_sen(k,j)=L_sen(k,j) -p1^2 /(2*p2*dt)- 0.5*log(2*pi*p2*dt);
      end
    end
end   
%3D-plot
meshc(ss,ssigma,L_sen);

% the original Guess for parameters
% assume sigma=0->a,g,s;then ->sigma
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


%Using Newton's rule to find MLE
Newton_N=0;
L_old=0;
L_new=0;
theta=theta_guess;
for i=1:N
    p1 = x(i+1) -x(i)+theta(1)*x(i)^3*dt +theta(2)*x(i)*dt;
    p2 = theta(3)^2 +theta(4)^2 *x(i)^2;
    L_new=L_new -p1^2 /(2*p2*dt)- 0.5*log(2*pi*p2*dt);
end

while abs(L_new-L_old)> 10^(-6)
    Newton_N = Newton_N+1;
    L_old=L_new;
    D=zeros(4,1);
    J=zeros(4,4);
    for i=1:N
        p1 = x(i+1) -x(i)+theta(1)*x(i)^3*dt +theta(2)*x(i)*dt;
        p2 = theta(3)^2 +theta(4)^2 *x(i)^2;
        
        D(1)=D(1) -x(i)^3 *p1 / p2;
        D(2)=D(2) -x(i)* p1 / p2;
        D(3)=D(3) +theta(3)*p1^2 / (p2^2*dt) - theta(3) / p2;
        D(4)=D(4) +theta(4)*x(i)^2*p1^2 / (p2^2*dt) - theta(4)*x(i)^2 / p2;
        
        J(1,1)=J(1,1) + x(i)^6*dt / p2;
        J(1,2)=J(1,2) + x(i)^4*dt / p2;
        
        J(1,3)=J(1,3) - 2*theta(3)*x(i)^3*p1 / (p2^2);
        J(1,4)=J(1,4) - 2*theta(4)*x(i)^5*p1 / (p2^2);
        J(2,2)=J(2,2) + x(i)^2*dt / p2;
        J(2,3)=J(2,3) - 2*theta(3)*x(i)*p1 / (p2^2);
        J(2,4)=J(2,4) - 2*theta(4)*x(i)^3*p1 / (p2^2);
        
        J(3,3)=J(3,3) + (3*theta(3)^2 -theta(4)^2*x(i)^2)*p1^2 / (p2^3*dt) +(-theta(3)^2+theta(4)^2*x(i)^2) / (p2^2);
        J(3,4)=J(3,4) + 4*theta(3)*theta(4)*x(i)^2*p1^2 / (p2^3*dt) -2*theta(3)*theta(4)*x(i)^2 / (p2^2);
        J(4,4)=J(4,4) - x(i)^2*(theta(3)^2 -3*theta(4)^2*x(i)^2)*p1^2 / (p2^3*dt) + x(i)^2*(theta(3)^2-theta(4)^2*x(i)^2) / (p2^2);   
    end
        J(2,1)=J(1,2);
        J(3,1)=J(1,3);
        J(4,1)=J(1,4);
        J(3,2)=J(2,3);
        J(4,2)=J(2,4);
        J(4,3)=J(3,4);
        theta=theta + inv(J)*D;

    L_new=0;
    for i=1:N
        p1 = x(i+1) -x(i)+theta(1)*x(i)^3*dt +theta(2)*x(i)*dt;
        p2 = theta(3)^2 +theta(4)^2 *x(i)^2;
        L_new=L_new-p1^2 /(2*p2*dt)- 0.5*log(2*pi*p2*dt);
    end
end
theta_sample(:,k)=theta;
% elseif (p==2) theta_sample_2(:,k)=theta;
%     else theta_sample_3(:,k)=theta;
%     end
%end
%end



