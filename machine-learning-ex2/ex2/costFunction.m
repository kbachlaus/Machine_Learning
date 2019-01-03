function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
z = X*theta;
Sig = sigmoid(z);
jtemp1 = log(Sig).*(-y);
jtemp2 = log(1-Sig).*(1-y) ;
J = (sum(jtemp1 - jtemp2))/m;

for j = 1:size(theta,1);
    sumgtemp=0;
    for i= 1:m;
        gtemp = (Sig(i)-y(i))*X(i,j);
        sumgtemp = sumgtemp + gtemp;
    grad(j) = sumgtemp/m;
    end
end
% for i = size(X,1)
%     jtemp1 = y(i)*(log(sigmoid(z)
%     jtemp2 = 
%     J = J + temp1 + temp2
%     gtemp1 =
%     grad
% end
% 

fprintf('Final J value: %f\n',J)

% =============================================================

end
