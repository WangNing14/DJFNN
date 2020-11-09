function mse = MSE( y_test, y_pre )

N = size(y_test,1);
Np = size(y_pre,1);

if N ~= Np
    disp('the number of test and prediction inconsistent');
    return;
end

mse =  sum( (y_test - y_pre).^2 ) / N /2;

end