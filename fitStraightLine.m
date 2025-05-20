function pFit = fitStraightLine(p1, p2, xlims)
% function fits to a straight line and extends it to xlims
% y = mX + b

xP = [p1(1) p2(1)];
yP = [p1(2) p2(2)];

% get slope and  intercept
m = (yP(2)-yP(1))/(xP(2)-xP(1));
b = yP(2) - (m *xP(2));

% set extended line
pFit(:,1) = xlims;
pFit(:,2) = (xlims*m) + b;
end