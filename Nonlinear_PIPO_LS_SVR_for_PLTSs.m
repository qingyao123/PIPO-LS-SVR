close all
clc
clear 
% ======================================================================================
% Proposed method：Build the two-stage nonlinear PIPO-LS-SVR model to achieve PLTS regression prediction directly.
% Logic: Combining the nuclear method with the temperature scaling technique, and using the original PLTS dataset for nonlinear regression prediction.
% ======================================================================================

%% Import the original PLTS dataset
INPUT = readmatrix('PLTS_dataset1 (Hotel Reviews Dataset).xlsx', 'Sheet', 'PLTS Input', 'Range', 'A1:Y34');  % PLTS inputs
OUTPUT = readmatrix('PLTS_dataset1 (Hotel Reviews Dataset).xlsx', 'Sheet', 'PLTS Output', 'Range', 'A1:E34'); % PLTS outputs

m = 5;      % The number of language terms in the PLTS
kfold = 5;  % 5-fold cross-validation

kernel = 1; % kernel function
% Nonlinear case: polynomial kernel(kernel = 1);Gaussian kernel(kernel = 2)

%% Model Parameter Optimization (Regularization Parameter C)
C_values = 1; % Regularization Parameter C
error_per_C = [];
for C_value = C_values
    [errors_list1,T1] = k_fold_cross_validation(INPUT, OUTPUT, kfold, C_value, m, kernel);
    [errors_list2,T2] = loocv(INPUT, OUTPUT, C_value, m, kernel);
    error_per_C = [error_per_C, mean(errors_list1,2), mean(errors_list2,2)];
end
error_per_C.'
% Show the errors
% line1: results of k-fold cross-validation 
% line2: results of LOO-CV

%% LOO-CV
function [results,T_list] = loocv(X, y, C_value, m, kernel)
    % LOO-CV
    T_list = [];
    [n, mxd] = size(X);
    d = mxd / m;
    weights_res = zeros(d,n);
    B_res = zeros(d,n);
    errors_list = zeros(6,n);
    for i = 1:n
        trainIdx = [1:i-1, i+1:n];
        %trainIdx
        valid = randperm(n-1, 10); % Take 10 samples as the validation set
        %valid
        validIdx = trainIdx(valid);
        trainIdx(valid) = [];  % Delete the selected index position
        testIdx = i;
        X_train = X(trainIdx,:);
        y_train = y(trainIdx,:);
        X_valid = X(validIdx,:);
        y_valid = y(validIdx,:);
        X_test = X(testIdx,:);
        y_test = y(testIdx,:);
        %% Train the linear PIPO-LS-SVR model
        [alpha, b_k] = PIPO_LS_SVR(X_train, y_train, C_value, m, kernel);   % Model Parameter Estimation
        valid_z = prediction(alpha, b_k, X_train, X_valid, m, kernel);      % Obtain the decision value of the prediction for the first stage
        [optimal_T,~] = temperature_scaling(valid_z, y_valid);              % Training temperature T
        pred_z = prediction(alpha, b_k, X_train, X_test, m, kernel);        % The decision values of the test set
        [~, ~, pred_y] = nll_loss(optimal_T, pred_z, y_test);
        T_list = [T_list, optimal_T];
        %% Model performance evaluation
        errors = get_errors(pred_y,y_test,m);                      % Calculate the prediction error of PLTS
        errors_list(:,i) = errors.';                               % Store the PLTS prediction errors of this experiment
    end
    results = errors_list;
end

function [errors_list,T_list] = k_fold_cross_validation(X, y, kfold, C_value, m, kernel)
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
    T_list = [];
    [N, mxd] = size(X);
    d = mxd / m;
    errors_list = zeros(6,kfold);

    %% Dividing the dataset: k-fold cross-validation method
    indices = crossvalind('KFold', N, kfold); % k-fold dataset partitioning
    for kf = 1:kfold
        test_mask = (indices == kf);          % Obtain the list of indices for dividing the dataset
        X_train = X(~test_mask, :);           % Inputs of PLTS training dataset
        y_train = y(~test_mask, :);           % Outputs of PLTS training dataset
        %X_train
        valid = randperm(size(X_train,1), 8); % The validation set
        X_valid = X_train(valid, :);          % Inputs of PLTS validation dataset
        y_valid = y_train(valid, :);          % Outputs of PLTS validation dataset
        X_train(valid, :) = [];
        y_train(valid, :) = []; 
        X_test = X(test_mask, :);             % Inputs of PLTS testing dataset
        y_test = y(test_mask, :);             % Outputs of PLTS testing dataset
        %X_train
        %X_valid
        %% Training two-stage nonlinear PIPO-LS-SVR model
        [alpha, b_k] = PIPO_LS_SVR(X_train, y_train, C_value, m, kernel);   % Model Parameter Estimation
        valid_z = prediction(alpha, b_k, X_train, X_valid, m, kernel);      % Obtain the decision value of the prediction for the first stage
        [optimal_T,~] = temperature_scaling(valid_z, y_valid);              % Training temperature T
        pred_z = prediction(alpha, b_k, X_train, X_test, m, kernel);        % The decision values of the test set
        [~, ~, pred_y] = nll_loss(optimal_T, pred_z, y_test);
        T_list = [T_list, optimal_T];
        %% Model performance evaluation
        errors = get_errors(pred_y,y_test,m);                        % Calculate the prediction error of PLTS
        errors_list(:,kf) = errors.';                                % Store the PLTS prediction errors of this experiment
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

function E_k = get_E_k(k, N, m)
    I_m_m = eye(m);
    vec_1_N = ones(N,1);
    E_k = vec_1_N * I_m_m(:,k).';
end

function K = kernels(L1, L2, kernel)
    [N1,~] = size(L1);
    [N2,~] = size(L2);

    if kernel == 1 % (1) Polynomial Kernel
        c = ones(N1,N2);
        d = 2; 
        K = (L1 * L2.' + c) .^ d;
    end
    if kernel == 2 % (2) Gaussian Kernel
        gamma = 10; 
        K = zeros(N1, N2);
        for i = 1:N1
            for j = 1:N2
                K(i,j) = exp(-gamma * sum((L1(i,:) - L2(j,:)).^2));
            end
        end
    end
end

function L_mat = generate_L_mat(input,m)
    [N,m_d] = size(input);
    d = m_d / m;
    % Extract the feature matrix for each dimension of the language term
    % L_k(k=1,2,...,m)=[x_k1,x_k2,...,x_kd]
    L_mat = cell(1,m);
    for k = 1:m
        L_k = zeros(N,d);
        for j = 1:d
            L_k(:,j) = input(:,m*(j-1)+k);
        end
        L_mat{k} = L_k;
    end
end

function [alpha, b_k] = PIPO_LS_SVR(input, y, C, m, kernel)
    [N, ~] = size(input);
    % vectors
    I_N_N = eye(N);
    vec_0_m = zeros(m,1);
    mat_0_m_m = zeros(m,m);

    % Extract the feature matrix for each dimension of the language term
    L_mat = generate_L_mat(input,m);

    %% Construct a system of linear equations
    % Coefficient matrix P
    P = [];
    % % The equation regarding the Lagrange multipliers alpha
    for k1 = 1:m
        line = [];
        for k2 = 1:m
            S = kernels(L_mat{k1}, L_mat{k2}, kernel);
            if k1 == k2
                S = (1/C) * I_N_N + S;
                line = [line, S];
            else
                line = [line, S];
            end
        end
        E_k = get_E_k(k1, N, m);
        line = [line, E_k];
        P = [P;line];
    end

    % The equations regarding B(k)
    line = [];
    for k = 1:m
        E_k = get_E_k(k, N, m).';
        line = [line, E_k];
    end
    line = [line, mat_0_m_m];
    P = [P;line];

    Y = [];
    for k = 1:m
        Y = [Y; y(:,k)];
    end
    q = [Y;vec_0_m];

    params = P\q;

    % Seperate alpha,b(k)
    alpha = zeros(N,m);
    for k = 1:m
        alpha(:,k) = params(N*(k-1)+1:N*k);
    end
    b_k = params((N*m+1):end);
end

function pred_y = prediction(alpha,b_k,trainX,testX,m,kernel)
    [N1,m_d1] = size(trainX); %Input matrix (Training data)
    [N2,m_d2] = size(testX);  %Input matrix (Testing data)
    if m_d1 == m_d2
        d = m_d1 / m;
    else
        print("The dimensions of the input matrices are inconsistent！");
    end
    d = m_d1 / m;
    L_mat1 = generate_L_mat(trainX,m);
    L_mat2 = generate_L_mat(testX,m);
  
    pred_y = zeros(N2,m); %pred_y
    for k = 1:m
        pred_y_k = zeros(N2,1);
        for h = 1:m
            alpha_mat = zeros(N1,N2);
            for n = 1:N2
                alpha_mat(:,n) = alpha(:,h);
            end
            k_mat = kernels(L_mat1{h}, L_mat2{k}, kernel);
            pred_y_k = pred_y_k + sum(alpha_mat .* k_mat).';
        end
        pred_y(:,k) = pred_y_k + b_k(k) * ones(N2,1);
    end
end

function [optimized_temp,cali_probs] = temperature_scaling(logits_val, probs_val)
    % Main function: Optimize temperature parameters
    % Input:
    %   logits_val - Logits of the validation set (matrix: [N, C])
    %   probs_val  - True probability distribution of the validation set (matrix: [N, C])
    % Output:
    %   optimized_temp - Optimized temperature value

    init_temp = 1.0;
    lr = 0.05;       % learning rate
    tol = 1e-6;      % Convergence tolerance
    max_iter = 100;  % maximum number of iterations

    % Optimize temperature parameters
    [optimized_temp, ~, cali_probs] = optimize_temperature_gd(logits_val, probs_val, ...
                                                  init_temp, lr, tol, max_iter);
    %fprintf('\nOptimized temperature: %.4f\n', optimized_temp);
end

function [optimized_temp, history, new_cali_probs] = optimize_temperature_gd(logits, probs, init_temp, lr, tol, max_iter)
    % Use the gradient descent method to optimize the temperature parameter
    % Input:
    %   logits     - Logits of the validation set
    %   probs      - True probability distribution
    %   init_temp  - Initial temperature value
    %   lr         - Learning rate
    %   tol        - Convergence tolerance
    %   max_iter   - Maximum number of iterations
    % Output:
    %   optimized_temp - Optimized temperature value
    %   history        - Optimization history record

    t = init_temp;
    history.temps = t;
    history.losses = [];

    % Compute the loss and gradient for the first time.
    [loss, grad, ~] = nll_loss(t, logits, probs);
    history.losses = [history.losses; loss];

    % fprintf('Iter 0: T = %.5f, Loss = %.5f, Grad = %.5e\n', t, loss, grad);

    % Gradient descent iteration
    for i = 1:max_iter
        % Gradient descent update (ensuring that the temperature is positive)
        new_temp = t - lr * grad;
        new_temp = max(new_temp, 1e-8);  % Prevent the temperature value from being non-positive

        % Calculate the loss and gradient for the new position
        [new_loss, new_grad, new_cali_probs] = nll_loss(new_temp, logits, probs);

        % Update the history record
        history.temps(end+1) = new_temp;
        history.losses(end+1) = new_loss;

        %fprintf('Iter %d: T = %.5f, Loss = %.5f, Grad = %.5e\n', i, new_temp, new_loss, new_grad);

        % Check the convergence condition (the gradient is close to zero)
        if abs(new_grad) < tol
            fprintf('Converged at iteration %d with gradient %.5e\n', i, new_grad);
            optimized_temp = new_temp;
            return;
        end

        % Update the current temperature and gradient
        t = new_temp;
        grad = new_grad;
    end

    %fprintf('Reached maximum iterations (%d)\n', max_iter);
    optimized_temp = t;
end

function [loss, grad, cali_probs] = nll_loss(temperature, logits, probs)
    % Calculate the negative log-likelihood loss and its gradient
    % Input:
    %   temperature - Temperature parameter
    %   logits      - Original logits
    %   probs       - True probability distribution
    % Output:
    %   loss - Negative log-likelihood loss
    %   grad - Gradient of the loss with respect to temperature

    [N, M] = size(logits);  % Obtain the sample size N and the number of categories M

    % Apply temperature scaling
    scaled_logits = logits / temperature;

    % Numerically stable log-sum-exp calculation
    max_vals = max(scaled_logits, [], 2);
    exp_vals = exp(scaled_logits - max_vals);
    log_sum_exp = log(sum(exp_vals, 2)) + max_vals;

    log_probs = scaled_logits - log_sum_exp;

    % Calculate the negative log-likelihood loss
    nll_i = sum(log_probs .* probs, 2);
    loss = -mean(nll_i);

    % Calculate the calibration probability
    cali_probs = exp(log_probs);

    % Calculate the expected logit value
    expected_logits = sum(cali_probs .* logits, 2);

    % Calculate the gradient components
    z_sum_zp = logits - expected_logits;
    grad_components = sum(probs .* z_sum_zp, 2) / (temperature^2);

    % Calculate the average gradient
    grad = mean(grad_components);
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
