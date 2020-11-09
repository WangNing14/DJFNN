% implementation of Disjunctive FNN function
% input: X:observed inputs, N x P
%        Y:observed output, N x 1
%        fuzzi:fuzzification of inputs, obtained based on fuzzification.m
%              fuzzi.input{1}.name    str
%              fuzzi.input1.MFsNum     R   
%              fuzzi.input1.MFsName    str
%              fuzzi.input1.MFsType    str ( trapmf )
%              fuzzi.input1.MFsPar     Matrix  ( MFsNum * 4 )
%        C: max number of rules 
%        Order:the order of consequance, less than 3
% output: model 

function model = DJFNN ( x_obs, y_obs, fuzzi, C, Order)

[N,Dim] = size(x_obs);
dim = fuzzi.input_number;
if dim ~= Dim
    disp('the dimention of fuzzi is not consistent with x_obs');
    return
end

%% normalization
% scale = max(x_obs) - min(x_obs);
% intercept = min(x_obs);
% x_obs = (x_obs - intercept)./scale;

scale = std(x_obs);intercept = mean(x_obs);
x_obs = (x_obs - repmat(intercept,N,1))./repmat(scale,N,1);

M = zeros(1,dim);
Par = cell(1,dim);
for i =  1:dim
    M(i) = fuzzi.input{i}.MFsNum;
    Par{i} = (fuzzi.input{i}.MFsPar - intercept(1,i))/scale(1,i);
    fuzzi.input{i}.MFsPar = Par{i};
end

%% define T(S)-norm
T_norm = @( a, b ) a .* b;
% S_norm = @( a, b ) a + b - a*b;
% T_norm = @( a, b ) min( a , b);
% S_norm = @( a, b ) max( a , b );
% T_norm = @( a, b ) b.*(a==1 & b~=1) + a.*(b==1 & a~=1) + 1.*(a==1 & b==1) + 0.*(a~=1 & b~=1);
S_norm = @( a, b ) b.*(a==0) + a.*(b==0) + 1.*(a~=0 & b~=0);

%% random searching Q{i} based on subdivisions by horizontal or vertical line
% initial definition
rmse_min = 10e10;
RMSE = zeros(C,1);
Q1 = cell(1,dim);
Q = Q1;
for i = 1 : dim
    Q1{i} = ones(M(i),1);
    Q{i} = [ones(M(i),1),zeros(M(i),C-1)];
end
RMSE(1) = myfun( x_obs,y_obs,fuzzi,1,Order, Q1 );
fun_num = 0;

%  search for subdivision along each dimention
for i = 2:C 
    sign = 0;
    for j = 1 : i-1                    % cut in the jth partition
        for l = 1 : dim                % cut along the lth dimension                    
            for position = 1 : M(l)-1  % cut at which position 
                Q_temp = Q;
                               
                for kk = 1 : dim
                    if kk == l
                        Q_temp{kk}(position+1:end,i) = Q_temp{kk}(position+1:end,j); 
                        Q_temp{kk}(position+1:end,j) = 0;
                    else
                        Q_temp{kk}(:,i) = Q_temp{kk}(:,j);
                    end
                end
                
                % check is Q_temp satisfy condition and calculate rmse
                if sum(Q_temp{l}(:,i))~=0 && sum(Q_temp{l}(:,j))~=0                                    
                    rmse = myfun( x_obs,y_obs,fuzzi,i,Order, Q_temp );             
                    fun_num = fun_num + 1;
                    if rmse < rmse_min
                        rmse_min = rmse;
                        Q_best = Q_temp;
                        sign = 1;
                    end
                else
                    continue
                end
                
            end
        end
    end                

    % determine optimal division Ap{1-i} and minimum RMSE in toal i clusters
    if sign == 0
        disp('more clusters does not increase accuracy')
        C = i;
        break
    end
    Q = Q_best;
    RMSE(i)= myfun( x_obs,y_obs,fuzzi, i, Order, Q );

end

%% cauculate consquence parameters matrix A
% B matrix, the degree of kth inputs belong to ith cluster
B = zeros(N,C);
for i = 1 : C
    s = zeros(N,dim);
    for l = 1 : dim
        for j = 1 : M(l)
            a = trapmf(x_obs(:,l),Par{l}(j,:));
            s_temp = T_norm(a,Q{l}(j,i));
            sq = S_norm(s_temp,s(:,l));
            s(:,l) = sq;
        end
    end
    s_temp = ones(N,1);
    for l = 1 : dim           
        B(:,i) = T_norm(s_temp,s(:,l));
        s_temp = B(:,i);
    end
end

% D matrix
if Order == 0
    D = 1./sum(B,2).*B;
else
    if Order == 1
        x_temp = [ones(N,1),x_obs];
    elseif Order == 2
        x_temp = [ones(N,1),x_obs,x_obs.*x_obs,x_obs(:,1).*x_obs(:,2)];
    elseif Order == 3
        x_temp = [ones(N,1),x_obs,x_obs.*x_obs,x_obs(:,1).*x_obs(:,2),...
                 x_obs.^3,x_obs(:,1).^2.*x_obs(:,2),x_obs(:,2).^2.*x_obs(:,1)];
    else
        disp('the order of consequence is larger than 3')
        return
    end      
    p = size(x_temp,2);
    D = ones(N,p*C);
    for l = 1 : p*C
        f = mod(l,p);
        if f == 0
            f = p;
            e = l/p;
        else
            e = (l-f)/p + 1;
        end
        D(:,l) = 1./sum(B,2).*B(:,e).*x_temp(:,f);
    end
end

% Moore-Penrose
A = pinv(D'*D)*D'*y_obs;

% stable-stete Kalman filter
% P0 = zeros(p*C,1);
% S0 = 1e8*eye(p*C);
% for k = 1 : N-1
%     S1 = S0 - (S0*D(k+1,:)'*D(k+1,:)*S0)/(1+D(k+1,:)*S0*D(k+1,:)');
%     P1 = P0 + S1*D(k+1,:)'*(y_obs(k+1)-D(k+1,:)*P0);
%     
%     S0 = S1;
%     P0 = P1;
% end
% A = P1;

% least absolute error estimate
% P = size(D,2);
% f = [zeros(1,P),ones(1,N)];
% Aa = [D,-speye(N);-D,-speye(N)];
% b = [y_obs;-y_obs];
% lb = [-inf*ones(1,P),zeros(1,N)];
% ub = [inf*ones(1,P),inf*ones(1,N)];
% options = optimoptions('linprog','Display','off');
% [x,fval]=linprog(f,Aa,b,[],[],lb,ub,options);
% A = x(1:P);

%% output
model.Q = Q;
model.A = A;
model.C = C;
model.Fuzzi = fuzzi;
model.minRMSE = rmse_min;
model.RMSE = RMSE;
model.scale = scale;
model.intercept = intercept;
model.Order = Order;
model.dim = dim;
model.fun_num = fun_num;
end
%% objective function and constrian function
function rmse = myfun( x_obs,y_obs,fuzzi, C, Order, Q )

[N,Dim] = size(x_obs);
dim_y = size(y_obs,2);
dim = fuzzi.input_number;
dim_Q = size(Q,2);
if dim ~= dim_Q
    disp('the dimention of fuzzi is not consistent with Q');
    return
end
if dim ~= Dim
    disp('the dimention of fuzzi is not consistent with x_obs');
    return
end

% parameter passing from fuzzi
M = zeros(1,dim);
Par = cell(1,dim);
for i =  1:dim
    M(i) = fuzzi.input{i}.MFsNum;
    Par{i} = fuzzi.input{i}.MFsPar;
end
    
% define T-norm and S-norm
T_norm = @( a, b ) a .* b;
% S_norm = @( a, b ) a + b - a*b;
% T_norm = @( a, b ) min( a , b);
% S_norm = @( a, b ) max( a , b );
% T_norm = @( a, b ) b.*(a==1 & b~=1) + a.*(b==1 & a~=1) + 1.*(a==1 & b==1) + 0.*(a~=1 & b~=1);
% S_norm = @( a, b ) b.*(a==0) + a.*(b==0) + 1.*(a~=0 & b~=0);

% B matrix, the degree of kth inputs belong to ith cluster
s = zeros(N,C,dim);
for l = 1 : dim
    for j = 1 : M(l)
        a = trapmf(x_obs(:,l),Par{l}(j,:));
        s_temp = T_norm(a,Q{l}(j,1:C));
        sq = s(:,:,l).*(s_temp==0) + s_temp.*(s(:,:,l)==0) + 1.*(s_temp~=0 & s(:,:,l)~=0);
%         sq = S_norm(s_temp,s(:,:,l));
        s(:,:,l) = sq;
    end
end
s_temp = ones(N,1);
for l = 1 : dim           
    B = T_norm(s_temp,s(:,:,l));
    s_temp = B;
end

% D matrix linear
if Order == 0
    D = 1./sum(B,2).*B;
else
    if Order == 1
        x_temp = [ones(N,1),x_obs];
    elseif Order == 2
        x_temp = [ones(N,1),x_obs,x_obs.*x_obs,x_obs(:,1).*x_obs(:,2)];
    elseif Order == 3
        x_temp = [ones(N,1),x_obs,x_obs.*x_obs,x_obs(:,1).*x_obs(:,2),...
                 x_obs.^3,x_obs(:,1).^2.*x_obs(:,2),x_obs(:,2).^2.*x_obs(:,1)];
    else
        disp('the order of consequence is larger than 3')
        return
    end      
    p = size(x_temp,2);
    D = ones(N,p*C);
    for l = 1 : p*C
        f = mod(l,p);
        if f == 0
            f = p;
            e = l/p;
        else
            e = (l-f)/p + 1;
        end
        D(:,l) = 1./sum(B,2).*B(:,e).*x_temp(:,f);
    end
end

% Moore-Penrose
A = pinv(D'*D)*D'*y_obs;
% stable-state Kalman filter
% P0 = zeros(p*C,1);
% S0 = 1e8*eye(p*C);
% for k = 1 : N-1
%     S1 = S0 - (S0*D(k+1,:)'*D(k+1,:)*S0)/(1+D(k+1,:)*S0*D(k+1,:)');
%     P1 = P0 + S1*D(k+1,:)'*(y_obs(k+1)-D(k+1,:)*P0);
%     
%     S0 = S1;
%     P0 = P1;
% end
% A = P1;

% least absolute error estimat
% P = size(D,2);
% f = [zeros(1,P),ones(1,N)];
% Aa = [D,-speye(N);-D,-speye(N)];
% b = [y_obs;-y_obs];
% lb = [-inf*ones(1,P),zeros(1,N)];
% ub = [inf*ones(1,P),inf*ones(1,N)];
% options = optimoptions('linprog','Display','off');
% [x,fval]=linprog(f,Aa,b,[],[],lb,ub,options);
% % A2 = x(1:p);
% rmse = fval/N;

y_pre = D*A;              
rmse = sqrt( sum(sum((y_obs - y_pre).^2))/N/dim_y );
% rmse = sum(abs(y_obs - y_pre))/N ;

end  









