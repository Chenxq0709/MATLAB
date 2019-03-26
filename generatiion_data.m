global N_slow
L = 100;
domain = 256;
N_window = 16; 
N_slow = fix(domain / N_window) ;
Trend = 1000;
dt_cal = 0.01;     %step length for discretization
store_freq = 5; % the storage dt =0.05
dt = dt_cal*store_freq;
store_time_start = 10;
N_theta = 19;

a= -1*[1;-2;1];
b= -0.5*[-1;-1;0;0;1;1];
c= 0.3*[5;3;0;-3;0;0;-10;-3;3;5];
s = 0.1;
sigma = 0.1;
%theta_i=[2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1];
theta_19=[a;b;c];

theta_true = zeros(N_theta+2,N_slow);
theta_true(1:19,:)= theta_19 *ones(1,N_slow);
theta_true(20, :)= s*ones(1,N_slow);
theta_true(21, :)= sigma*ones(1,N_slow);
    
%Milstein method: approximate numerical solution
% initialaztion is corresponding to the FB Code
para = theta_true;
% para= para_MLE;
 N_cal = Trend/dt_cal;
 x_cal= zeros(N_slow, N_cal );
 
 N_x = (Trend-store_time_start) / dt;   % number of discretization
 x=zeros(N_slow, N_x);

 m=0;
 for i = 1: N_slow 
    initial = 0;
  for j =1: N_window
    m = j+(i-1)*N_window;
    m = L/(N_window * N_slow)*(m-1);
    initial = initial +sin(2*pi*m/L) +sin(4*pi*m/L);
  end
  
  x_cal(i,1) = initial / N_window;
end
 
store_count=0;
start_count= false;
x_count=0;
 for t = 1: N_cal
  for i= 1:N_slow
    k1 = integral_ODE(x_cal(:,t) , i, para(1:N_theta,i), dt_cal);
    k2 = integral_ODE(x_cal(:,t) + 0.5*k1 , i, para(1:N_theta,i), dt_cal);
    k3 = integral_ODE(x_cal(:,t) + 0.75*k2 , i, para(1:N_theta,i), dt_cal);
    
    x_cal(i,t+1) = x_cal(i,t) + (2*k1 + 3*k2 + 4*k3)/9; 
    
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

function y=integral_ODE(input,i_th,theta,dt)
global N_slow
 if i_th==1
        xx = [input(N_slow),input(i_th),input(i_th+1)];
     elseif i_th==N_slow
        xx = [input(i_th-1),input(i_th),input(1)]; 
     else 
        xx = [input(i_th-1),input(i_th),input(i_th+1)];
 end
    y=(dot(theta(1:3)', xx)+ dot(theta(4:9)', two_variables(xx)) + dot(theta(10:19)',  three_variables(xx))) *dt;
return
end