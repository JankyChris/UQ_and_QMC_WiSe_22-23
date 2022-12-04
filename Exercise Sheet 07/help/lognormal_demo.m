% Written by Vesa Kaarnioja, 2022
% Monte Carlo for PDE with lognormal diffusion coefficient (lecture 6)

% Precompute the FEM data
level = 5;
[mass,grad,nodes,element,interior,centers,ncoord,nelem] = FEMdata(level);

% Set up the lognormally parameterized random diffusion coefficient
s = 100; % stochastic dimension
decay = 1.1; % the "decay parameter" \vartheta from the slides
indices = (1:s)';
deterministic = indices.^(-decay) .* sin(pi*indices*centers(:,1)') .* sin(pi* indices * centers(:,2)');
a = @(y) exp(deterministic'*y);

% Use n == 2^maxiter as the "reference solution"
maxiter = 20;

% Initialize the source term and loading vector
f = @(x) x(:,1); % Source term, in this case f(x) = x_1
rhs = mass(interior,:)*f(nodes); % Loading term; not affected by random diffusion coefficient

% Initialize the parallel pool (uses the Parallel Computing Toolbox)
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

sums = [];
means = [];
null = zeros(ncoord,1);

% Range over an increasing number of random samples
for ii = 0:maxiter % n = 2^ii, ii = 0,...,maxiter
    disp(['iteration: ',num2str(ii)]);
    n = max(1,2^(ii-1));
    tmp = zeros(ncoord,1);
    % The main loop
    parfor k = 1:n
        mcnode = randn(s,1); % A realization of the normally distributed parametric vector "y"
        A = UpdateStiffness(grad,a(mcnode)); % Assemble the stiffness matrix corresponding to a(y), y == mcnode
        sol = null;
        sol(interior) = A(interior,interior)\rhs; % Solve PDE
        tmp = tmp + sol; % Sum over Monte Carlo points
    end
    sums = [sums,tmp]; % Store the sum
    means = [means,sum(sums(:,1:size(sums,2)),2)/n]; % Compute sample average (this is the Monte Carlo estimate)
end

% Use the solution corresponding to n = 2^maxiter as the reference solution
ref = means(:,end);

% Compute the L2 errors of solutions corresponding to n = 2^ii, 
% ii = 0,...,maxiter-1, vis-a-vis the reference solution
% (For the computation of the L2 norm of FE function using the
% mass matrix, see exercise 2 of week 5!)
errors = [];
for ii = 1:size(means,2)-1
    errors = [errors;2^(ii-1),sqrt((means(:,ii)-ref)'*mass*(means(:,ii)-ref))];
end

% Find the least squares fit for the errors
lsq = [ones(size(errors(:,1))),log(errors(:,1))]\log(errors(:,2));
lsq(1) = exp(lsq(1));

% Plot the L2 errors as a log-log plot.
figure;
loglog(errors(:,1),errors(:,2),'r.','MarkerSize',18), hold on
x = 2.^(0:maxiter-1);
loglog(x,lsq(1)*x.^(lsq(2)),'b','LineWidth',2), hold off
set(gca,'FontSize',14,'FontWeight','bold')
legend('data',['slope: ',num2str(lsq(2))])
title('Lognormal example (Monte Carlo)')
xlabel('n')
ylabel('L^2 error')
