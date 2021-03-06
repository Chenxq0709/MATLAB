% parameters setup for  dx=(A x +B xx + C xxx) dt +s dw1 +sigma dw2
L=100;
Domain = 256;
N_window = 16; 
N_slow = 16;

A = ones(N_slow, 3); %paramenters for the one-multiplication part 
B = ones(N_slow, 6); %paramenters for the two-multiplication part
C = ones(N_slow, 9); %paramenters for the three-multiplication part
N_theta=18;
s = ones(N_slow, 1);
sigma = ones(N_slow, 1);
Trend = 100;

dt=0.01;     %step length for discretization
N_time_step= Trend /dt +1;   % number of discretization
N_sample=1;  %number of samples
N_theta=20*N_slow;   %size of parameters collection

theta_true=[A;B;C;s;sigma];
theta_sample=zeros(N_theta,N_sample);  %space for storage MLE of samples

%MLE for N_sample samples
for m=1:N_sample
   x=generating_SDE(A, B, C, s, sigma, N_slow, Trend, dt);  
   para_guess=initial_guess(x);
  for i= 1: N_slow
     options = optimoptions('fminunc','Display','iter-detailed','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective');
  
     [para_sample(:,i,m),MLE] = fminunc(@negative_log_likelihood(i,), para_guess(:,i).',options);

%     sens_plot(x,theta_guess,theta_sample(:,m));
end
%  [sample_mean, sample_rel_err]= sample_statistics(theta_sample_1_2,theta_true);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x=generating_SDE(A, B, C, s, sigma, N_slow, Trend, dt)
%Milstein method: approximate numerical solution
% initialaztion is corresponding to the FB Code
 x=zeros(N_slow,Trend);
 for i = 1: N_slow 
  for j =1: N_window
    m = L/ domain * ((i-1)*N_window +j) ;
    x_initial = x_initial +m;
  end
  x(i,1) = x_initial / n_window;
 end

 N_time_step= Trend /dt +1;

 for t = 1: N_time_step 
  for i= 1:N_slow

    dw1 = sqrt(dt)*randn;
    dw2 = sqrt(dt)*randn;

     if i==1
        xx = [x(t,N_slow),x(t, i),x(t, i+1)];
     elseif i==N_slow
        xx = [x(t, i-1),x(t, i),x(t, 1)]; 
     else 
        xx = [x(t, i-1),x(t, i),x(t, i+1)];
     end

    x(t+1,i) = x(t,i) + dot(A(i,:), xx) *dt;
    x(t+1,i) = x(t+1,i) + dot(B(i,:), two_variables(xx)) *dt;
    x(t+1,i) = x(t+1,i) + dot(C(i,:),  three_variables(xx)) *dt;
    x(t+1,i) = x(t+1,i) + s(i)*dw1 + sigma(i)*x(t,i)*dw2 + 0.5*sigma(i)^2*x(t,i)*(dw2^2-dt);
  end
 end
return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y=two_variables(input)
%size of y =6
%size of input =3, input=[x_i-1,x_i,x_i+1]

y = [];
for i = 1:3
    for j = i:3
        y = [y, input(i)*input(j)];
    end
end 
return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y=three_variables(input)
%size of y=9
%size of input =3, input=[x_i-1,x_i,x_i+1]

y = [];
for i = 1:3
    for j= i:3
       for k = j:3
         y = [y , input(i)*input(j)*input(k)];
       end
    end
end 
return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function para_guess= initial_guess(x) 

dt= evalin('base','dt');
N_theta = evalin('base','N_theta');
N_slow = evalin('base','N_slow');
N_time_step = evalin('base','N_time_step');

para_guess=zeros(N_theta+2, N_slow);

 for i= 1: N_slow

    Y_Y = zeros(N_theta,N_theta);
    Y_dx = zeros(N_theta,1);

    for t= 1:N_time_step

         if i==1
            xx = [x(t,N_slow),x(t, i),x(t, i+1)];
         elseif i==N_slow
            xx = [x(t, i-1),x(t, i),x(t, 1)]; 
         else 
            xx = [x(t, i-1),x(t, i),x(t, i+1)];
         end

      Y_t=[xx,two_variables(xx),three_variables(xx)];
      Y_Y = Y_Y + Y_t'* Y_t;
      Y_dx= Y_dx - (x(t+1,i)-x(t,i))*  Y_t';

    end

    Y_Y = Y_Y * dt;
    para_guess([1:N_theta],i)=inv(Y_Y) * Y_dx;

    p1=0;
    for t= 1:N_time_step
         
         if i==1
            xx = [x(t,N_slow),x(t, i),x(t, i+1)];
         elseif i==N_slow
            xx = [x(t, i-1),x(t, i),x(t, 1)]; 
         else 
            xx = [x(t, i-1),x(t, i),x(t, i+1)];
         end

      Y_t=[xx,two_variables(xx),three_variables(xx)];
      p1 = p1 + (x(t+1,i) - x(t,i) + dot(para_guess([1:N_theta],i),Y_t)*dt)^2;
    end

    para_guess(19,i)=  sqrt(p1/(N_time_step *dt));
    para_guess(20,i)= sqrt((p1-para_guess(19,i)^2*dt)/Y_Y(2,2));
  end
return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f,G,H] = negative_log_likelihood(i_th, para)
%i_th slow varible

dt= evalin('base','dt');
N_theta = evalin('base','N_theta');
N_slow = evalin('base','N_slow');
N_time_step = evalin('base','N_time_step');
x = evalin('base','x');

f=0;  % negative log-likelihood function, which we want to minimize
G=zeros(N_theta+2, 1);  %gradient 
H=zeros(N_theta+2, N_theta+2);  %hessian

   for t= 1:N_time_stpe
        
     if i_th==1
        xx = [x(t,N_slow),x(t, i_th),x(t, i_th+1)];
     elseif i_th==N_slow
        xx = [x(t, i_th-1),x(t, i_th),x(t, 1)]; 
     else 
        xx = [x(t, i_th-1),x(t, i_th),x(t, i_th+1)];
     end

     Y_t=[xx,two_variables(xx),three_variables(xx)];

     p = x(t+1,i_th)-x(t,i_th)- dot(para([1:N_theta]), Y_t)*dt;
     q = para(N_theta+1)^2 +para(N_theta+2)^2 *x(t, i_th)^2;
        
     f=f + p^2 /(2*q*dt) + 0.5*log(2*pi*q*dt);
     
     G(1:N_theta)= G(1:N_theta) + Y_t'* (p/q) ;
     G(N_theta+1)= G(N_theta+1)- (p^2 / (q^2*dt) - 1/ q)* para(N_theta+1);
     G(N_theta+2)= G(N_theta+2)- (p^2 / (q^2*dt) - 1/ q)* para(N_theta+2)*x(t,i_th)^2;
     
     H(1:N_theta,1:N_theta)=  H(1:N_theta,1:N_theta) + Y_t'* Y_t *(dt/q);
     H(1,N_theta+1)=  H(1,N_theta+1) -2*para(N_theta+1)/(q^2)* Y_t';
     H(1,N_theta+2)=  H(1,N_theta+2) -2*para(N_theta+2)/(q^2)*x(t,i_th)^2* Y_t';
        
     H(N_theta+1,N_theta+1)=H(N_theta+1,N_theta+1) + (3*para(N_theta+1)^2 -para(N_theta+2)^2*x(t,i_th)^2)*p^2 / (q^3*dt) +(-para(N_theta+1)^2+para(N_theta+2)^2*x(t,i_th)^2) / (q^2);
     H(N_theta+1,N_theta+2)=H(N_theta+1,N_theta+2) + 4*para(N_theta+1)*para(N_theta+2)*x(t,i_th)^2*p^2 / (q^3*dt) -2*para(N_theta+1)*para(N_theta+2)*x(t,i_th)^2 / (q^2);
     H(N_theta+2,N_theta+2)=H(N_theta+2,N_theta+2) - x(t,i_th)^2*(para(N_theta+1)^2 -3*para(N_theta+2)^2*x(t,i_th)^2)*p^2 / (q^3*dt) + x(t,i_th)^2*(para(N_theta+1)^2-para(N_theta+1)^2*x(t,i_th)^2) / (q^2);   
    end
     
     H(N_theta+1,1)=H(1,N_theta+1)';
     H(N_theta+2,1)=H(1,N_theta+2)';
     H(N_theta+2,N_theta+1)=H(N_theta+1,N_theta+2);

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
