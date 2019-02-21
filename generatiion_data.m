
L = 100;
domain = 256;
N_window = 16; 
N_slow = fix(domain / N_window) ;
Trend = 100;
dt = 0.01;     %step length for discretization
store_freq = 5; % the storage dt =0.05
store_time_star = 10;
N_theta = 19;

a= [1;-2;1];
b= [-1;-1;0;0;1;1];
c=0.1*[5;3;0;-3;0;0;-10;-3;3;5];
s = 1;
sigma = 1;
theta_i=[1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2];
theta_19=[a;b;c];

theta_true = zeros(N_theta+2,N_slow);
theta_true(1:19,:)= theta_19 .* theta_i;
theta_true(20, :)= s*ones(1,N_slow);
theta_true(21, :)= sigma*ones(1,N_slow);
    
%Milstein method: approximate numerical solution
% initialaztion is corresponding to the FB Code
 para = theta_true;
 N_cal = Trend/dt;
 x_cal= zeros(N_slow, N_cal );
 
 x_dt = dt* store_freq;
 N_x = (Trend-store_time_star) / x_dt;   % number of discretization
 x=zeros(N_slow, N_x);

 m=0;
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