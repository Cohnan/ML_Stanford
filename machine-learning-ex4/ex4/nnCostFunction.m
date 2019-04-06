function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%hyp = predict(Theta1, Theta2, X);
%display(size(hyp));
%display(y(1));
%imagesc(reshape(X(1,:), 20, 20))
% What I learned from playing with this: y is a mx1 matrix where each entry
% is a number from 1 to 10 corresponding to the labels 1 through 0, and so is hyp

z2s = [ones(m, 1) X] * Theta1';
a2s = sigmoid(z2s);

z3s = [ones(m, 1) a2s] * Theta2';
a3s = sigmoid(z3s);

newy   = zeros(m, num_labels);

for i = 1:m
  newy(i, y(i)) = 1; 
  %newhyp(i, hyp(i)) = 1;s
endfor

 
J = sum(  (-1/m * ( newy.*(log(a3s)) + (1-newy).*(log(1-a3s)) )) (:)  );

% Regularization
J += lambda/(2*m) * sum( ( Theta1(:, 2:end).*Theta1(:, 2:end) )(:) );
J += lambda/(2*m) * sum( ( Theta2(:, 2:end).*Theta2(:, 2:end) )(:) );


%% Backpropagation!!

for t = 1:m
  % Step 1
  a1 = [1; X(t,:)'];
  z2 = z2s(t,:)';
  a2 = [1; a2s(t,:)'];
  z3 = z3s(t,:)';
  a3 = a3s(t,:)';
  
  % Step 2: delta for output layer
  del3 = a3 - newy(t, :)'; % I had written newy(t)', I suffered for 1:30 hours!!
  
  % Step 3: delta for hidden layers
  del2 = (Theta2(:, 2:end)'*del3) .* sigmoidGradient(z2);
  
  % Step 4: Accumulate the gradient from this training example
  
  Theta1_grad += del2 * a1';
  Theta2_grad += del3 * a2';
  
endfor

Theta1_grad /= m;
Theta2_grad /= m;

Theta1_grad(:, 2:end) += lambda/m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) += lambda/m * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
