%% parameters setup for  dx=(A x +B xx + C xxx) dt +s dw1 +sigma dw2
global  N_slow N_theta dt
%parameters set up
N_theta = 19;
L = 100;
domain = 256;
N_window = 16; 
N_slow = fix(domain / N_window) ;
Trend = 500;
dt_cal = 0.001;     %step length for discretization
store_freq = 5; % the storage dt =0.05
dt = dt_cal*store_freq;  %step length fdiscretizationor 
store_time_start = 20;
time = store_time_start:dt:Trend;
a=[1;-2.1;1];
b= -10*[-1;-1;0;0;1;1];
c= 1*[5;3;0;-3;0;0;-10;-3;3;5];
s = 0.1;
sigma = 0.05;
theta_19=[a;b;c];

theta_true = zeros(N_theta+2,N_slow);
theta_true(1:19,:)= theta_19 *ones(1,N_slow);
theta_true(20, :)= s*ones(1,N_slow);
theta_true(21, :)= sigma*ones(1,N_slow);
%main part
x_test = generation_data(theta_true, Trend, dt_cal, store_freq, store_time_start);
energy = plot_energy(time,x_test);
%x_test = x_005(:,1:40001);
x = x_test;
% three initial guess options:
%para_initial= theta_true*0.9;
para_initial=initial_guess(x);
%para_initial= 0.01*randn(N_theta+2,N_slow);
 para_MLE=zeros(N_theta+2,N_slow);  %space for storage MLE result
  for i= 1: N_slow
     initial = para_initial(:,i);
     options = optimoptions('fminunc','Display','iter-detailed','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','MaxIter', 5000, 'OptimalityTolerance', 1e-10);
     [para_MLE(:,i),] = fminunc(@negative_log_likelihood, initial ,options);
  end
x_MLE= generation_data(para_MLE, Trend, dt_cal, store_freq, store_time_start);
plot_autocorr(x_test,x_MLE);
stat_true = zeros(16,4);
stat_MLE = zeros(16,4);
for i= 1:16
    stat_true(i,1) = mean(x_test(i,:));
    stat_MLE(i,1) = mean (x_MLE(i,:));
    stat_true(i,2) = var(x_test(i,:));
    stat_MLE(i,2) = var(x_MLE(i,:));
    stat_true(i,3) = moment(x_test(i,:)',3);
    stat_MLE(i,3) = moment(x_MLE(i,:)',3);
    stat_true(i,4) = moment(x_test(i,:)',4);
    stat_MLE(i,4) = moment(x_MLE(i,:)',4);
end
mean(stat_true)
mean(stat_MLE)


%  para_MLE=zeros(N_theta+2,N_slow);  %space for storage MLE result
%   for i= 1: N_slow
%      initial = para_initial(:,i);
%      options = optimoptions('fminunc','Display','iter-detailed','Algorithm','trust-region','SpecifyObjectiveGradient',true,'MaxIter', 5000);
%      [para_MLE(:,i),] = fminunc(@likelihood, initial ,options);
%   end
% x_MLE= generation_data(para_MLE, Trend, dt_cal, store_freq, store_time_start);
% plot_autocorr(x_test,x_MLE);

function energy=plot_energy(time,y)
global N_slow
N_time_step = size(y,2); 
energy=zeros(1,N_time_step);
for i=1:N_time_step
    for j=1:N_slow
    energy(i)= energy(i)+y(j,i)^2;
    end
end
plot(time,energy);
xlabel('Time');
ylabel('Energy');
return;
end

function plot_autocorr(x_test,x_MLE)
%plot correlation 
Corr_test= auto_correlation_fun(x_test);
Corr_MLE = auto_correlation_fun(x_MLE);
xaxes=0:1:200;
plot(xaxes,mean(Corr_test),xaxes,mean(Corr_MLE));
axis([0 200 0 1]);
ylabel('auto correlation');
xlabel('t lag');
legend('true','MLE');
end

function Corr=auto_correlation_fun(x)
N = size(x,1); % number of discretization
T = size(x,2);
lag_time= 200;

 Corr=zeros(N,lag_time+1);
  for tau =1:lag_time+1
    t_lag= T - tau;
    s=0;
    v=0;
    for j= 1:N
       m=mean(x(j,:));
      for  i=1:t_lag
        s = s + (x(j,i)-m)*(x(j,i+tau-1)-m);
        v = v + (x(j,i)-m)^2;
      end
       Corr(j,tau)=s/v;
   end
   
  end
return
end

 function x_0 = generation_initial(N_slow)
 L=100;
 N_window=16;
 m=0;
 for i = 1: N_slow 
    initial = 0;
  for j =1: N_window
    m = j+(i-1)*N_window;
    m = L/(N_window * N_slow)*(m-1);
    initial = initial +sin(2*pi*m/L) +sin(4*pi*m/L);
  end
  
  x_0 = initial / N_window;
 end
return
 end

 function x =generation_data(para, Trend, dt_cal, store_freq, store_time_start)
 global  N_slow N_theta
 
 N_cal = Trend/dt_cal;
 x_cal= zeros(N_slow, N_cal );
% x_cal(:,1) = generation_initial(N_slow); 
x_cal(:,1)= ones(N_slow, 1);
 dt = dt_cal*store_freq;
 N_x = (Trend-store_time_start) / dt;   % number of discretization
 x=zeros(N_slow, N_x);
 
 store_count=0;
 start_count= false;
 x_count=0;
 for t = 1: N_cal
  for i= 1:N_slow
    k1 = integral_ODE(x_cal(:,t) , i, para(1:N_theta,i));
    k2 = integral_ODE(x_cal(:,t) + dt_cal*0.5*k1 , i, para(1:N_theta,i));
    k3 = integral_ODE(x_cal(:,t) + dt_cal*(-k1 + 2*k2) , i, para(1:N_theta,i));
    
    x_cal(i,t+1) = x_cal(i,t) + (1*k1 + 4*k2 + 1*k3)/6 *dt_cal; 
    
    dw1 = sqrt(dt_cal)*randn;
    dw2 = sqrt(dt_cal)*randn;
    x_cal(i,t+1) = x_cal(i,t+1) + para(20,i)*dw1 + para(21,i)*x_cal(i,t)*dw2 + 0.5*para(21,i)^2*x_cal(i,t)*(dw2^2-dt_cal);
  end
  
  if t*dt_cal >=  store_time_start
      start_count= true;
  end
     
  store_count= store_count +1;
  if (store_count == store_freq) 
      store_count = 0;
    if start_count== true
      x_count = x_count +1;
      x(:,x_count) = x_cal(:,t+1);
    end  
  end
 end
 return
 end
 
function y=integral_ODE(input,i_th,theta)
global N_slow
 if i_th==1
        xx = [input(N_slow),input(i_th),input(i_th+1)];
     elseif i_th==N_slow
        xx = [input(i_th-1),input(i_th),input(1)]; 
     else 
        xx = [input(i_th-1),input(i_th),input(i_th+1)];
 end
    y=(dot(theta(1:3)', xx)+ dot(theta(4:9)', two_variables(xx)) + dot(theta(10:19)',  three_variables(xx)));
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
global N_slow N_theta dt

para_guess=zeros(N_theta+2, N_slow);
N_time_step = size(x,2);   % number of discretization

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

      Y_t=[xx,two_variables(xx),three_variables(xx)]';
      Y_Y = Y_Y + Y_t* Y_t' *dt;
      Y_dx= Y_dx + (x(i,t+1)-x(i,t))*  Y_t;

    end

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
global  N_slow N_theta  dt
%i_th slow varible
i_th= evalin('base','i');
x= evalin('base','x');
N_time_step = size(x,2);   % number of discretization

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

     Y_t=[xx,two_variables(xx),three_variables(xx)]';

     p = x(i_th,t+1) - x(i_th,t) -  para(1:N_theta)' *Y_t*dt;
     q = para(N_theta+1)^2 + para(N_theta+2)^2 *x(i_th,t)^2;
        
     f = f + p^2 /(2*q*dt) + 0.5*log(2*pi*q*dt);
     
     G(1:N_theta)= G(1:N_theta) - (p/q)*Y_t ;
     m1 = (-p^2 / (q^2*dt) + 1/ q);
     G(N_theta+1)= G(N_theta+1)+ m1 * para(N_theta+1);
     G(N_theta+2)= G(N_theta+2)+ m1 * para(N_theta+2)*x(i_th,t)^2;
     
     H(1:N_theta,1:N_theta)=  H(1:N_theta,1:N_theta) + Y_t* Y_t' *(dt/q);
     m2 = 2*p/(q^2);
     H(1:N_theta,N_theta+1)=  H(1:N_theta,N_theta+1) + m2* para(N_theta+1)* Y_t;
     H(1:N_theta,N_theta+2)=  H(1:N_theta,N_theta+2) + m2* para(N_theta+2)*x(i_th,t)^2 * Y_t;
     
     m3 =  (2*p^2) / (q^3*dt) - 1/q^2;   
     H(N_theta+1,N_theta+1)=H(N_theta+1,N_theta+1) + m3*2*para(N_theta+1)^2 + m1;                                     
     H(N_theta+1,N_theta+2)=H(N_theta+1,N_theta+2) + m3* 2*para(N_theta+1)*para(N_theta+2)*x(i_th,t)^2;
     H(N_theta+2,N_theta+2)=H(N_theta+2,N_theta+2) + m3* 2*para(N_theta+2)^2*x(i_th,t)^4 + m1* x(i_th,t)^2;
      
    end
     
     H(N_theta+1,1:N_theta)=H(1:N_theta,N_theta+1)';
     H(N_theta+2,1:N_theta)=H(1:N_theta,N_theta+2)';
     H(N_theta+2,N_theta+1)=H(N_theta+1,N_theta+2);

end

function [f,G] =likelihood(para)
global  N_slow N_theta  dt
%i_th slow varible
i_th= evalin('base','i');
x= evalin('base','x');
N_time_step = size(x,2);   % number of discretization

f=0;  % negative log-likelihood function, which we want to minimize
G=zeros(N_theta+2, 1);  %gradient 
   for t= 1:N_time_step-1
        
     if i_th==1
        xx = [x(N_slow,t),x(i_th,t),x(i_th+1,t)];
     elseif i_th==N_slow
        xx = [x(i_th-1,t),x(i_th,t),x(1,t)]; 
     else 
        xx = [x(i_th-1,t),x(i_th,t),x(i_th+1,t)];
     end

     Y_t=[xx,two_variables(xx),three_variables(xx)]';

     p = x(i_th,t+1) - x(i_th,t) -  para(1:N_theta)' *Y_t*dt;
     q = para(N_theta+1)^2 + para(N_theta+2)^2 *x(i_th,t)^2;
        
     f = f + p^2 /(2*q*dt) + 0.5*log(2*pi*q*dt);
     
     G(1:N_theta)= G(1:N_theta) - (p/q)*Y_t ;
     m1 = (-p^2 / (q^2*dt) + 1/ q);
     G(N_theta+1)= G(N_theta+1)+ m1 * para(N_theta+1);
     G(N_theta+2)= G(N_theta+2)+ m1 * para(N_theta+2)*x(i_th,t)^2;     
   end
end