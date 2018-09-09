function [X_norm, mu, sigma] = featureNormalize(X)
  m = length(X(:,1));
  n = length(X(1,:));
  X_norm = zeros(m,n);
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
  mu = mean(X);
  sigma = std(X);
  for j = 1:n
    for i = 1:m
      X_norm(i,j) = (X(i,j) - mu(j)) / sigma(j);
    end
  end
% ============================================================
end
