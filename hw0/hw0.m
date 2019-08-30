% find the critical points
syms b1 b2 real;
f = (b1^2 + b2 -11)^2 + (b1 +b2^2 -7)^2;
dx = diff(f,b1);
dy = diff(f,b2);
ddx = diff(dx,b1);
sol = vpasolve([dx dy]);
solution = [double(sol.b1) double(sol.b2)];

% find the hessian matrix and substitue b1,b2's values in the matrix
hess = det(hessian(f,[b1,b2]));
b1 = double(sol.b1);
b2 = double(sol.b2);
hessdet = double(subs(hess));
ddx = double(subs(ddx));

% plot a contour plot and indicate the critical points
fcontour(f,[-5 5 -5 5]);
FF = subs(f,sol)
% fsurf(f,[-5 5 -5 5]);
hold on
scatter3(sol.b1,sol.b2,FF,30,'r+')



for n=1:9
    if hessdet(n) > 0
        if ddx(n)>0
            text(sol.b1(n),sol.b2(n),FF(n),"minimum",'Color', 'black','FontSize',20);
        elseif ddx(n)<0
            text(sol.b1(n),sol.b2(n),FF(n),"maximum",'Color', 'black','FontSize',20);
        end
    else
        text(sol.b1(n),sol.b2(n),FF(n),"saddle point",'Color', 'black','FontSize',20);
    end
end







