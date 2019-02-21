

function Corr=correlation_plot(x)
N = size(x,1); % number of discretization
T = size(x,2);
lag_time= 20;

 Corr=zeros(lag_time+1,1);
  for tau =1:lag_time+1
    t_lag= T - tau;
    m=0;
    for j= 1:N
      for  i=1:t_lag
        m = m + x(j,i)*x(j,i+tau-1);
      end
   end
   Corr(tau) = m / (N*t_lag);
  end
return
end

% Tiny is super cute!