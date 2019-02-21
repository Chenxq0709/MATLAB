%% parameters setup for  dx=(A x +B xx + C xxx) dt +s dw1 +sigma dw2
global  N_slow N_theta N_time_step dt

N_slow = 16;
N_theta = 19;

a= [1,-2,1];
A = repmat(a,N_slow,1); %paramenters for the one-multiplication part 
b= [-1,-1,0,0,1,1];
B = repmat(b,N_slow,1); %paramenters for the two-multiplication part
c=0.1*[5,3,0,-3,0,0,-10,-3,3,5];
C = repmat(c,N_slow,1); %paramenters for the three-multiplication part
s = ones(N_slow, 1);
sigma = ones(N_slow, 1);
para_true=[A,B,C,s,sigma]';
%x= generating_data(para_true);


N_time_step = size(x,2);   % number of discretization
dt = 0.05;     %step length for discretization

para_MLE=zeros(N_theta+2,N_slow);  %space for storage MLE of samples

%%MLE for samples
%for m=1:N_sample

   para_guess=initial_guess(x);
  for i= 1: N_slow
     initial = para_guess(:,i);
     options = optimoptions('fminunc','Display','iter-detailed','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective');
     [para_MLE(:,i),MLE] = fminunc(@negative_log_likelihood, initial ,options);

%     sens_plot(x,theta_guess,theta_sample(:,m));
  end
%  [sample_mean, sample_rel_err]= sample_statistics(theta_sample_1_2,theta_true);

%%
function x = generating_data(para)
L = 100;
domain = 256;
N_window = 16; 
N_slow = fix(domain / N_window) ;
Trend = 100;
dt = 0.01;     %step length for discretization
store_freq = 5; % the storage dt =0.05
store_time_star = 10;


%Milstein method: approximate numerical solution
% initialaztion is corresponding to the FB Code
 N_cal = Trend/dt;
 x_cal= zeros(N_slow, N_cal );
 
 x_dt = dt* store_freq;
 N_x = (Trend-store_time_star) / x_dt;   % number of discretization
 x=zeros(N_slow, N_x);
 
 for i = 1: N_slow 
    initial = 0;
  for j =1: N_window
    m = L/ domain * ((i-1)*N_window +j) ;
    initial = initial +sin(2*pi*m/L) +sin(4*pi*m/L);
  end
  
  x_cal(i,1) = initial / N_window;
 end
 
store_count=0;
start_count= false;
x_count=0;
 for t = 1: N_cal
  for i= 1:N_slow

    dw1 = sqrt(dt)*randn;
    dw2 = sqrt(dt)*randn;

     if i==1
        xx = [x_cal(N_slow,t),x_cal(i, t),x_cal(i+1, t)];
     elseif i==N_slow
        xx = [x_cal(i-1,t),x_cal(i,t),x_cal(1,t)]; 
     else 
        xx = [x_cal(i-1,t),x_cal(i,t),x_cal(i+1,t)];
     end

    x_cal(i,t+1) = x_cal(i,t) + dot(para(1:3,i)', xx) *dt;
    x_cal(i,t+1) = x_cal(i,t+1) + dot(para(4:9,i)', two_variables(xx)) *dt;
    x_cal(i,t+1) = x_cal(i,t+1) + dot(para(10:19,i)',  three_variables(xx)) *dt;
    x_cal(i,t+1) = x_cal(i,t+1) + para(20,i)*dw1 + para(21,i)*x_cal(i,t)*dw2 + 0.5*para(21,i)^2*x_cal(i,t)*(dw2^2-dt);
  end
  
  if t*dt >  store_time_star
      start_count= true;
  end
     
  store_count= store_count +1;
  if (store_count == store_freq) 
      store_count = 0;
    if start_count== true
      x_count = x_count +1;
      x(1:N_slow,x_count) = x_cal(1:N_slow,t+1);
    end  
  end
  
 end
 return
end 
%% 
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
%%

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
%%

function para_guess= initial_guess(x) 
global N_slow N_theta N_time_step dt

para_guess=zeros(N_theta+2, N_slow);

 for i= 1: N_slow

    Y_Y = zeros(N_theta,N_theta);
    Y_dx = zeros(N_theta,1);

    for t= 1:N_time_step-1

         if i==1
            xx = [x(N_slow,t),x(i,t),x(i+1,t)];
         elseif i==N_slow
            xx = [x(i-1,t),x(i,t),x(1,t)]; 
         else 
            xx = [x(i-1,t),x(i,t),x(i+1,t)];
         end

      Y_t=[xx,two_variables(xx),three_variables(xx)];
      Y_Y = Y_Y + Y_t* Y_t';
      Y_dx= Y_dx + (x(i,t+1)-x(i,t))*  Y_t';

    end

    Y_Y = Y_Y * dt;
    opts.SYM = true;
    para_guess(1:N_theta,i)= linsolve(Y_Y,Y_dx);

    p1=0;
    for t= 1:N_time_step-1
         
         if i==1
            xx = [x(N_slow,t),x(i,t),x(i+1,t)];
         elseif i==N_slow
            xx = [x(i-1,t),x(i,t),x(1,t)]; 
         else 
            xx = [x(i-1,t),x(i,t),x(i+1,t)];
         end

      Y_t=[xx,two_variables(xx),three_variables(xx)];
      p1 = p1 + (x(i,t+1) - x(i,t) - dot(para_guess(1:N_theta,i),Y_t)*dt)^2;
    end

    para_guess(N_theta+1,i)=  sqrt(p1/(N_time_step *dt));
    para_guess(N_theta+2,i)=  sqrt((p1-para_guess(N_theta+1,i)^2*dt)/Y_Y(2,2));
 end
  
return
end
%%

function [f,G,H] = negative_log_likelihood(para)
global L  N_window N_slow N_theta Trend N_time_step dt
%i_th slow varible
i_th= evalin('base','i');
x= evalin('base','x');

f=0;  % negative log-likelihood function, which we want to minimize
G=zeros(N_theta+2, 1);  %gradient 
H=zeros(N_theta+2, N_theta+2);  %hessian

   for t= 1:N_time_step-1
        
     if i_th==1
        xx = [x(N_slow,t),x(i_th,t),x(i_th+1,t)];
     elseif i_th==N_slow
        xx = [x(i_th-1,t),x(i_th,t),x(1,t)]; 
     else 
        xx = [x(i_th-1,t),x(i_th,t),x(i_th+1,t)];
     end

     Y_t=[xx,two_variables(xx),three_variables(xx)];

     p = x(i_th,t+1)-x(i_th,t)- dot(para(1:N_theta), Y_t) *dt;
     q = para(N_theta+1)^2 +para(N_theta+2)^2 *x(i_th,t)^2;
        
     f=f + p^2 /(2*q*dt) + 0.5*log(2*pi*q*dt);
     
     G(1:N_theta)= G(1:N_theta) - Y_t'* (p/q) ;
     m1 = (-p^2 / (q^2*dt) + 1/ q);
     G(N_theta+1)= G(N_theta+1)+ m1 * para(N_theta+1);
     G(N_theta+2)= G(N_theta+2)+ m1 * para(N_theta+2)*x(i_th,t)^2;
     
     H(1:N_theta,1:N_theta)=  H(1:N_theta,1:N_theta) + Y_t'* Y_t .*(dt/q);
     m2 = 2*p/(q^2);
     H(1:N_theta,N_theta+1)=  H(1:N_theta,N_theta+1) + m2* para(N_theta+1)* Y_t';
     H(1:N_theta,N_theta+2)=  H(1:N_theta,N_theta+2) + m2* para(N_theta+2)*x(i_th,t)^2 * Y_t';
         
     H(N_theta+1,N_theta+1)=H(N_theta+1,N_theta+1) +  (4*p^2/(q^3*dt)-1/q^2)*2*para(N_theta+1)^2 + m1;
     %p^2*(4* para(N_theta+1)^2 - q) / (q^3*dt) + (-2*para(N_theta+1)^2 + q) / (q^2)
     m3 =  (2*p^2) / (q^3*dt) - 1/q^2;                                              
     H(N_theta+1,N_theta+2)=H(N_theta+1,N_theta+2) + 2*para(N_theta+1)*para(N_theta+2)*x(i_th,t)^2* m3 ;
     H(N_theta+2,N_theta+2)=H(N_theta+2,N_theta+2) + m3*2*para(N_theta+2)^2*x(i_th,t)^4 + m1* x(i_th,t)^2;
     %x(i_th,t)^2*p^2* (4*para(N_theta+2)^2*x(i_th,t)^2- q)/ (q^3*dt) + x(i_th,t)^2*(-2*para(N_theta+2)^2*x(i_th,t)^2 + q) / (q^2);   
    end
     
     H(N_theta+1,1:N_theta)=H(1:N_theta,N_theta+1)';
     H(N_theta+2,1:N_theta)=H(1:N_theta,N_theta+2)';
     H(N_theta+2,N_theta+1)=H(N_theta+1,N_theta+2);

end