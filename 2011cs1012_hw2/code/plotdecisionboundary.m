function plotdecisionboundary(w, X, y,deg)
%plotdecisionboundary Plots the data points X and y into a new figure with
%deg is the degree of the polynolial used for featuretransform
%the decision boundary defined by w
% Plot Data
plotdata(X(:,2:3), y);
hold on
% Here is the grid range
x1 = linspace(0, 10, 50);
x2 = linspace(0, 10, 50);
z = zeros(length(x1),length(x2));
% Evaluate z = w*x over the grid
for i = 1:length(x1)
        for j = 1:length(x2)
            z(i,j) = mapFeature(x1(i), x2(j),deg)*w;
        end
end
z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
contour(x1,x2, z, [0,0], 'LineWidth', 2);
hold off

end
