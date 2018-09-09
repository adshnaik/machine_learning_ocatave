function [J] = computeCostMulti(X, y, theta) 
    n = length(X(1,:));
    for i = 1:1:n
		  prediction = X(:,i) * theta(i)
      squareerror = (prediction - y).^ 2
      J(i) = 1 / (2 * m) * sum(squareerror); 
    end
end