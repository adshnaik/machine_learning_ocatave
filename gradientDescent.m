function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% Initialize some useful values
  m = length(y); % number of training examples
  n = length(X(1,:))
  error = zeros(n,1);
  temp = 0;
  J_history = zeros(num_iters, 1);
  J = zeros(num_iters, 1);
  
  for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    prediction = X * theta;
    error = X' * (prediction - y);
    theta = theta - alpha * 1 / m * error;
    
    % ============================================================

    % Save the cost J in every iteration    
    J(iter) = computeCost(X, y, theta);
  end
  J_history = J;
end
