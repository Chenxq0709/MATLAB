%plot correlation 
Corr_test= auto_correlation_fun(x_test);
Corr_MLE = auto_correlation_fun(x_MLE);
xaxes=0:1:20;
plot(xaxes,mean(Corr_test),xaxes,mean(Corr_MLE));
legend('true','MLE');
%cal statistics:mean,var,3rd-moment,4th-moment
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

function Corr=auto_correlation_fun(x)
N = size(x,1); % number of discretization
T = size(x,2);
lag_time= 20;

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
