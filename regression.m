function [betas, resid] = regression(data,regressors,alpha,constant)
%    Taku Ito
%    2/21/2019
%
%    OLS regression using closed form equation: betas = (X'X + alpha*I)^(-1) X'y
%    Set alpha = 0 for regular OLS.
%    Set alpha > 0 for ridge penalty
%
%    PARAMETERS:
%        data = observation x feature matrix (e.g., time x regions)
%        regressors = observation x feature matrix
%        Keyord arguments:
%        alpha = regularization term. 0 for regular multiple regression. >0 for ridge penalty
%        constant = true/false - pad regressors with 1s
%
%    OUTPUT
%        betas = coefficients X n target variables
%        resid = observations X n target variables
    
    % Add 'constant' regressor
    if constant==true
        one_arr = ones(size(regressors,1),1);
        X = [one_arr regressors];
    end

    % construct regularization term
    LAMBDA = eye(size(X,2))*alpha;

    % Least squares minimization
    C_ss_inv = pinv(X'*X + LAMBDA);
    
    betas = C_ss_inv * (X'*data);
    %Calculate residuals
    resid = data - betas(1,:) + X(:,2:end)*betas(2:end,:);

end
