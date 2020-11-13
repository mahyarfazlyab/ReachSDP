function output = reach_sdp_poly(net, input_set, c, repeated, sys, verbose)
% function output = reach_sdp_poly(net, input_set, c, repeated, sys, verbose)
%   Reach-SDP for DNNs with ReLU activation, polytopic sets
%
% ----- How to use this function -----
%
% Inputs:
%   net       - neural network parameters
%   input_set - input set of initial states
%   c         - orientation vector of the facet of the output set
%   repeated  - flag indicating whether to consider repeated nonlinearity
%   sys       - system information including dynamics and constraints
%   verbose   - flag indicating whether to report solving status
% 
% Authors: Haimin Hu, Mahyar Fazlyab
% Last modified by Haimin Hu (2020.11.13)


%% Initialization
% number of hidden layers (activation layers)
num_layers = length(net.dims)-2;

% state dimension
dim_x = net.dims(1);

% control dimension
dim_u = net.dims(end);

% number of neurons in the last hidden layer (the projector)
dim_last_hidden = dim_u;

% total number of neurons (including the projector, excluding x0)
num_neurons = sum(net.dims(2:end-1)) + 2*dim_u;

% dimension of the SDP problem / M matrix
dim_sdp = dim_x + num_neurons + 1;

% Input set
X_in = input_set.set;
input_type = input_set.type;

% Box outer-approximation of the input polytope
X_in_box = X_in.outerApprox;
x_max = X_in_box.b(1:2);
x_min = -X_in_box.b(3:4);

% Preactivation calculated using interval arithmatic
[Y_min,Y_max,X_min,X_max,~,~] = interval_arithmatic(net,x_min,x_max);
X_min = [X_min; zeros(2*dim_u,1)];
X_max = [X_max; zeros(2*dim_u,1)];
Y_min = [Y_min; zeros(2*dim_u,1)];
Y_max = [Y_max; zeros(2*dim_u,1)];
I_pos = intersect(find(Y_min>0),find(Y_max>0));
I_neg = intersect(find(Y_min<0),find(Y_max<0));

% DNN parameters of the last layer
Wout = net.weights{end};
bout = net.biases{end}(:);
Wout = double(Wout);
bout = double(bout);

% ReLU slope-restricted nonlinearity 
alpha = 0;
beta = 1;

% System matrices
As = sys.A;
Bs = sys.B;

% Setting up CVX
if(verbose)
    cvx_begin sdp
else
    cvx_begin sdp quiet
end

cvx_solver mosek


%% Input Quadratic
if strcmp(input_type,'polytope')
    % polytopic input set
    dim_gam = size(X_in.A,1);
    variable Gam(dim_gam,dim_gam) symmetric
    P = [X_in.A'*Gam*X_in.A -X_in.A'*Gam*X_in.b; -X_in.b'*Gam*X_in.A X_in.b'*Gam*X_in.b];
elseif strcmp(input_type,'box')
    % box input set
    variable tau(dim_x,1) nonnegative
    P = [-2*diag(tau) diag(tau)*(x_min+x_max);(x_min+x_max).'*diag(tau) -2*x_min.'*diag(tau)*x_max];
else
    disp('Error: unsupported input set!');
    output = [];
    return
end

% lifting matrix Ein
Ein = zeros(dim_x+1,dim_sdp);
Ein(1:dim_x,1:dim_x) = eye(dim_x);
Ein(end,end) = 1;

Min = Ein'*P*Ein;


%% Output Quadratic
variable b
S = [zeros(dim_x,dim_x) c; c' -2*b];
% lifting matrix Eout
cs = zeros(dim_x,1);
Eout = [As zeros(dim_x,num_neurons-dim_last_hidden) Bs];
Eout = [Eout cs; zeros(1,size(Eout,2)) 1];

Mout = Eout'*S*Eout;


%% QC for ReLU activation functions
% repeated nonlinearity
T = zeros(num_neurons);
if(repeated)
    if ~isempty(X_min)
        II = eye(num_neurons);
        C = [];
        if(numel(I_pos)>1)
            C = nchoosek(I_pos,2);
        end
        if(numel(I_neg)>1)
            C = [C;nchoosek(I_neg,2)];
        end
        m = size(C,1);
        if(m>0)
            variable zeta(m,1)
            E = II(:,C(:,1))-II(:,C(:,2));
            T = E*diag(zeta)*E';
        end
    else
        II = eye(num_neurons);
        C = nchoosek(1:num_neurons,2);
        m = size(C,1);
        if(m>0)
            variable zeta(m,1) nonnegative
            E = II(:,C(:,1))-II(:,C(:,2));
            T = E*diag(zeta)*E';
        end
    end
end

variable nu(num_neurons,1) nonnegative
variable lambda(num_neurons,1)
variable eta(num_neurons,1) nonnegative
variable D_diag(num_neurons,1) nonnegative

% bounded nonlinearity 
Dt = diag(D_diag);
D = blkdiag(eye(num_neurons-2*dim_u),0*eye(2*dim_u))*Dt*blkdiag(eye(num_neurons-2*dim_u),0*eye(2*dim_u));

Q11 = -2*alpha*beta*(diag(lambda)+T);
Q12 = (alpha+beta)*(diag(lambda)+T);
Q13 = -nu;
Q21 = Q12.';
Q22 = -2*(diag(lambda))-2*T-(D+D.');
Q23 = eta+nu+D*X_min+D.'*X_max;
Q31 = Q13.';
Q32 = Q23.';
Q33 = 0-X_min'*D.'*X_max-X_max'*D*X_min;

Q = [Q11 Q12 Q13; Q21 Q22 Q23; Q31 Q32 Q33];

% lifting matrix for Q
A = [];
aa = [];
for i=1:num_layers % 0 to l-1
    A = blkdiag(A, net.weights{i});
    aa = [aa; net.biases{i}(:)];
end
A  = blkdiag(A, Wout, -eye(dim_u));
A  = [A zeros(size(A,1),dim_u)];
aa = [aa; bout-sys.ulb; sys.uub];
B  = [zeros(num_neurons,dim_x) blkdiag(eye(num_neurons-dim_u), -eye(dim_u))];
bb = [zeros(num_neurons-2*dim_u,1); -sys.ulb; sys.uub];
A = double(A);
B = double(B);
aa = double(aa);
bb = double(bb);
Emid = [A aa; B bb; zeros(1,size(B,2)) 1];

Mmid = Emid'*Q*Emid;

clear Q11 Q12 Q13 Q21 Q22 Q23 Q31 Q32 Q33;


%% Formulate and solve the SDP

minimize b

subject to

Min + Mmid + Mout<=0;

if strcmp(input_type, 'polytope')
    Gam(:)>=0;
end

cvx_end

isSolved = strcmp(cvx_status,'Inaccurate/Solved') || strcmp(cvx_status,'Solved');

isFeasible = isSolved;

message = ['time: ', num2str(cvx_cputime), '| solved: ', num2str(isSolved), '| bound: ', num2str(b) ];
if verbose
    disp(message);
end

output.cpu_time = cvx_cputime;
output.bound = b;
output.solver_status = cvx_status;
output.isFeasible = isFeasible;
end