function [J] = computeCost(X, y, theta)
% Initialize some useful values
 m = length(y); % number of training examples
 error = zeros(size(y)); 
 
%%Return this values
  J = 0; 
% ====================== YOUR CODE HERE ======================
 error = (X * theta - y).^ 2;
 J = 1 / (2 * m) * sum(error); 

% ============================================================
end
