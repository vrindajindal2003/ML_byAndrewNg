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
%s=size(X)
X=[ones(m,1) X];
%size(X)
z2=Theta1*X';
%size(z2)

a2=sigmoid(z2);
%size(a2)
a2=[ones(1,m);a2];
%size(a2)
z3=Theta2*a2;
%size(z3)
a3=sigmoid(z3);

%a3=size(a3)
o=zeros(num_labels,m);
%oupt=size(o)
y(y==0)=10;
for i=1:m
    o(y(i),i)=1;
end
%oupt=size(o)

lhx=log(a3);
omlx=log(1-a3);
%sl=size(lhx)
%so=size(omlx)
%hxt=size(a3)
int1=o.*lhx+(1-o).*omlx;
J=sum(sum(int1),2)/(-m);
T1=Theta1;
T1(:,1)=[];
T2=Theta2;
T2(:,1)=[];
J=J+(lambda/(2*m))*(sum(sum(T1.*T1),2)+sum(sum(T2.*T2),2));




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
for t=1:m
    xt=(X(t,:))';
    a1=xt;
    %sizea1=size(a1)
    z2=Theta1*xt;
    %sizez2=size(z2)
    a2=sigmoid(z2);
    %sa2=size(a2)
    a2=[1;a2];
    %sa2=size(a2)
    z3=Theta2*a2;
    %sz3=size(z3)
    a3=sigmoid(z3);
    %sz3=size(z3);
    yp=zeros(num_labels,1);
    %syp=size(yp)
    yp(y(t))=1;
    delta3=a3-yp;
    %st2=size(Theta2)
    %sdelta3=size(delta3)
    %delta3=delta3(2:end);
    delta2=(Theta2'*delta3).*a2.*(1-a2);
    %delta2=[0;delta2];
    %delta3=[0;delta3];
    %%t1g=size(Theta2_grad)
    %d2=size(delta3)
    %sxt=size(a2)
    delta2=delta2(2:end);
    
    %delta3=delta3(2:end);
    %d2=size(delta3)
    
    Theta1_grad=Theta1_grad+delta2*(xt');
    Theta2_grad=Theta2_grad+delta3*(a2');
end
Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;
    


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

tta1=Theta1;
tta2=Theta2;
tta1(:,1)=zeros(size(Theta1,1),1);
tta2(:,1)=zeros(size(Theta2,1),1);
Theta1_grad=Theta1_grad+(lambda/m)*(tta1);
Theta2_grad=Theta2_grad+(lambda/m)*(tta2);

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
