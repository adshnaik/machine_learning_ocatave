function [theta,J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
% Initialize some useful values
  error = zeros(n,1);
	m = length(y); % number of training examples
	J_history = zeros(num_iters,1);
  	
  for iter = 1:1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    error = X' * (X * theta - y);
    theta = theta - alpha * 1 / m * error;
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
  end
end