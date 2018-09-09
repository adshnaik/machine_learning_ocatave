%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization
%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);



% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) ,y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

%%=================part 0: Compute cost Multi=====================
function [J] = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
	      m = length(y); % number of training examples
       
% You need to return the following variables correctly 
	      J = 0;
	      prediction = 0;
	      squareerror = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
	      
		     prediction = X * theta;
		     squareerror = (prediction - y).^ 2;
		     J = 1 / (2 * m) * sum(squareerror); 
	      
end
%%End of computeCostMulti=========================================
%% ================ Part 1: Feature Normalization ================
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
      X_norm(i,j) = [X(i,j) - mu(j)] / sigma(j);
    end
  end
% ============================================================
end




% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X,mu,sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1),X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
function [theta,J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
  val = num_iters;
  temp = 0;
  prediction = 0;
  error = 0;
  n = length(X(1,:));
	m = length(y); % number of training examples
	J_history = zeros(num_iters,1);
  %theta2 = zeros(n,num_iters);
  %theta3 = zeros(n,num_iters);
	
  for iter = 1:1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the valuess
    %       of the cost function (computeCostMu
    %       of the cost function (computeCostMulti) and gradient here.
      
      prediction = X * theta;
      for j = 1 : n
        for i = 1 : m
          error(i,j) = (prediction(i) - y(i)) * X(i,j);
        end 
      end
      for i = 1:n
        temp = theta(i) - alpha * 1 / m * sum(error(:,i));
        theta(i) = temp;
      end
      
    % ============================================================
      %for k = 1:n
       % theta2(k,iter) = theta(k);
      %end
    % Save the cost J in every iteration    
      J_history(iter) = computeCostMulti(X, y, theta);
  end
  
  %theta3 = theta2(:);
  %theta =theta2(:,val);
end

%END GradientDescent==============================================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 500;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);


% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this
X2 = input('Enter the size of house in sq-ft::');
X3 = input('Enter number of Bedroom:: ');
X2_norm = (X2 - mu(1)) / sigma(1);
x3_norm = (X3 - mu(2)) / sigma(2);
X0_norm = [1, X2_norm, x3_norm]; 
price = X0_norm * theta; 

% ============================================================

fprintf(['Predicted price of a {%0.0f sq-ft, %0.0f br} house ' ...
         '(using gradient descent):\n $%0.2f\n'],X2,X3,price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.
  theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------
  X_inv = pinv(X' * X);
  theta =  X_inv * X' * y;

end
%END of Normal equation======================================

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
[theta] = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this
X0 = [1, X2, X3]; 
price = X0 * theta; 


% ============================================================

fprintf(['Predicted price of a {%0.0f sq-ft, %0.0f br} house ' ...
         '(using normal equations):\n $%0.2f\n'],X2,X3, price);

