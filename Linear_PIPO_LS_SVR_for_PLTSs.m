close all
clc
clear
% ======================================================================================
% Proposed method：Build the linear PIPO-LS-SVR model to achieve PLTS regression prediction directly.
% Logic: Based on the averaging operator of distribution assessments, use the original PLTS dataset to build a linear PIPO-LS-SVR model to achieve linear regression.
% ======================================================================================

%% Import the original PLTS dataset
INPUT = readmatrix('PLTS_dataset1 (Hotel Reviews Dataset).xlsx', 'Sheet', 'X', 'Range', 'A1:Y34');
OUTPUT = readmatrix('PLTS_dataset1 (Hotel Reviews Dataset).xlsx', 'Sheet', 'Y', 'Range', 'A1:E34');
m = 5;      % The number of language terms in the PLTS
kfold = 5;  % 5-fold cross-validation

%% Model Parameter Optimization (Regularization Parameter C)
C_values = [0.5,0.7,1,1.5,2,5,10,20,50,100,1000]; % Regularization Parameter C

error_per_C = [];
for C_value = C_values
    [errors_list1, weights_res1, B_res1] = k_fold_cross_validation(INPUT, OUTPUT, kfold, C_value, m);
    [errors_list2, weights_res2, B_res2] = loocv(INPUT, OUTPUT, C_value, m);
    error_per_C = [error_per_C, mean(errors_list1,2), mean(errors_list2,2)];
end

function [results, weights_res, B_res]= loocv(X, y, C_value, m)
    % LOO-CV
    [n, mxd] = size(X);
    d = mxd / m;
    weights_res = zeros(d,n);
    B_res = zeros(m,n);
    errors_list = zeros(6,n);
    for i = 1:n
        trainIdx = [1:i-1, i+1:n];
        testIdx = i;
        X_train = X(trainIdx,:);
        y_train = y(trainIdx,:);
        X_test = X(testIdx,:);
        y_test = y(testIdx,:);
        %% Train the linear PIPO-LS-SVR model
        [weights, B] = LSSVR_linear_QP(X_train, y_train, C_value, m);
        weights_res(:,i) = weights;
        B_res(:,i) = B;
        pred_y = prediction(X_test, weights, B, m);  % Model Parameter Estimation
        
        %% Model performance evaluation
        errors = get_errors(pred_y,y_test,m);  % Calculate the prediction error of PLTS
        errors_list(:,i) = errors.';                               
    end
    results = errors_list;
end

function [errors_list, weights_res, B_res] = k_fold_cross_validation(X, y, kfold, C_value, m)
    % =====================================================================
    % Function: Divide the training set and test set using the k-fold cross-validation method, take the average error of k experiments to evaluate the model performance
    % Input:
    %   INPUT - PLTS input matrix [N, (m*d)]
    %   OUTPUT - PLTS output vector [N, m]
    %   kfold - k-fold
    %   C_value - regularization parameter (C)
    %   m - number of language terms
    % Output:
    %   errors_list - prediction error matrix on the test set [6 types of errors * kfold]
    % =====================================================================
    [N, mxd] = size(X);
    d = mxd / m;
    errors_list = zeros(6,kfold);
    weights_res = zeros(d,kfold);
    B_res = zeros(m,kfold);

    %% Dividing the dataset: k-fold cross-validation method
    indices = crossvalind('KFold', N, kfold);
    for kf = 1:kfold
        test_mask = (indices == kf);
        X_train = X(~test_mask, :);
        y_train = y(~test_mask, :);
        X_test = X(test_mask, :);
        y_test = y(test_mask, :);

        %% Train the linear PIPO-LS-SVR model
        [weights, B] = LSSVR_linear_QP(X_train, y_train, C_value, m);
        weights_res(:,kf) = weights;
        B_res(:,kf) = B;
        pred_y = prediction(X_test, weights, B, m);
        
        errors = get_errors(pred_y,y_test,m);
        errors_list(:,kf) = errors.';
    end
end

function L_X = get_L_X(X, m)
% Extract the feature matrix for each dimension of the language term L_X(k)=[x(k)1,x(k)2,...,x(k)d]
    N = size(X, 1);
    d = size(X, 2) / m;
    L_X = cell(1,m);
    for k = 1:m
        L_X_k = zeros(N,d);
        for j = 1:d
            L_X_k(:,j) = X(:,m*(j-1)+k);
        end
        L_X{k} = L_X_k;
    end
end

function [w, b] = LSSVR_linear_QP(X, y, C_value, m)
    % PIPO-LS-SVR
    N = size(X,1);
    d = size(X,2) / m;

    % Extract the feature matrix for each dimension of the language term
    L_X = get_L_X(X, m);
    L_y = get_L_X(y, m);

    %% QP solver
    H = zeros(d+m, d+m);
    D1 = zeros(d, d);
    for k = 1:m
        D1 = D1 + L_X{k}.' * L_X{k};
    end
    H(1:d, 1:d) = eye(d) + C_value * D1;

    for k = 1:m
        H(1:d, d+k) = C_value * sum(L_X{k}, 1)';
    end

    for k = 1:m
        H(d+k, 1:d) = C_value * sum(L_X{k}, 1);   % ones(1,N)*X
    end

    H(d+1:d+m, d+1:d+m) = C_value * N * eye(m);
    H = (H + H') / 2;

    f = zeros(d+m, 1);
    D2 = zeros(d, 1);
    for k = 1:m
        D2 = D2 + L_X{k}.' * L_y{k};
        f(d+k) = -C_value * sum(L_y{k}); % -C * 1'y(k)
    end
    f(1:d) = -C_value * D2; % -C * X(k)'y(k)

    %% constraints
    A = [-eye(d), zeros(d, m)];
    b_ineq = zeros(d, 1);

    Aeq = [ones(1,d), zeros(1,m);zeros(1,d), ones(1,m)];
    beq = [1;0];

    lb = [];
    ub = [];
    
    %% QP
    options = optimoptions('quadprog', ...
        'Display', 'off', ...
        'Algorithm', 'interior-point-convex');
    
    [v_opt, fval, exitflag] = quadprog(H, f, A, b_ineq, Aeq, beq, lb, ub, [], options);

    if exitflag <= 0
        warning('quadprog未成功收敛，退出标志: %d', exitflag);
    end
    
end

function weights = get_weights(L_mat, gamma, theta, N, m, d)
    % Estimated value of the regression coefficient
    weights = zeros(1,d);
    theta_vec = theta * ones(1,d);
    for k = 1:m
        gamma_k = gamma(((k-1)*N+1):k*N);
        weights = weights + gamma_k'*L_mat{k};
    end
    weights = (weights - theta_vec)';
end

function pred_y = prediction(X, weights, B, m)
    % Fitting values
    [N, mxd] = size(X);
    d = mxd / m;
    X_mat = cell(1,d); 
    for j = 1:d
        first = (j-1)*m+1;
        last = j*m;
        X_mat{j} = X(:,first:last);
    end
    pred_y = X_mat{1} * weights(1);
    for j = 2:d
        pred_y = pred_y + X_mat{j} * weights(j);
    end
    pred_y = pred_y + ones(N,1)*B';
    pred_y = get_norm_y(pred_y);
end

function norm_y = get_norm_y(output_y)
    % Standardize the predicted output
    [n, m] = size(output_y);
    norm_y = zeros(n, m);
    for i = 1:n
        for k = 1:m
            if output_y(i,k) <= 0
                output_y(i,k) = 0.0000000000000000001;
            end
        end
        norm_y(i,:) = output_y(i,:) / sum(output_y(i,:));
    end
end

function errors = get_errors(pred_y,real_y,m)
    % =====================================================================
    % Function: calculate PLTS Prediction Error
    % Input:
    %   pred_y - Predicted output (PLTS)
    %   real_y - True output (PLTS)
    % Output:
    %   errors - Five types of prediction errors
    % =====================================================================
    MAE = mean(sum(abs(pred_y - real_y),2));             % Mean absolute error (MAE)
    MSE = mean(sum((pred_y - real_y).^2,2));             % Mean squared error (MSE)
    RMSE = sqrt(MSE);                                    % Root mean squared error (RMSE)
    diff_mat = (pred_y - real_y).^2;   
    mm = [];   
    for k = 1:m   
        mm = [mm,diff_mat(:,k) * (k^2)];   
    end   
    L_RMSE = sqrt(mean(sum(mm,2)./(m^3)));               % Root mean squared error based on the PLTS distance (L_RMSE)
    real_y = get_norm_y(real_y);
    KL = real_y .* (log(real_y ./ pred_y));   
    %KL
    MKLD = mean(sum(KL,2));                              % Mean Kullback–Leibler Divergence (MKLD)
    KL1 = real_y .* (log((2 .* real_y) ./ (real_y + pred_y)));
    %KL1
    KL2 = pred_y .* (log((2 .* pred_y) ./ (real_y + pred_y)));
    %KL2
    MJSD = mean((sum(KL1,2) ./ 2) + (sum(KL2,2) ./ 2));  % Mean Jensen–Shannon Divergence (MJSD)

    errors = [MAE,MSE,RMSE,L_RMSE,MKLD,MJSD];
end