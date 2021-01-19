function [bound,time,status] =  reach_sdp(net,Asys,Bsys,input_polytope,c,options)

% Author: Haimin Hu & Mahyar Fazlyab
% email: haiminh@princeton.edu mahyarfazlyab@jhu.edu
% Last revision: December 2020

version = '20.12';

%%------- FUNCTION DECRIPTION -----------

% Inputs:
%   net            - nnsequential object representing the neural network
%   input_polytope - input_polytope = {x | H*[x;-1] <= 0}
%   c              - matrix of normal vectors of the facets of the
%                    output set: c(:,i) is the i-th normal vector
%   Asys           - state transition matrix of the LTI system
%   Bsys           - input matrix of the LTI system
%   options        - solver options with fields solver, verbose, repeated
%                       - solver = 'mosek','sedumi','sdpt3' (choose supported solvers for YALMIP)
%                       - verbose = 0 or 1 
%                       - repeated = 0 or 1 (indicating whether to consider
%                       repeated nonlinearity)
% Outputs:
%   bound          - minimal value of b such that c'*f(x)<=b for all x in
%                    input_polytope

%%------------- BEGIN CODE --------------

solver = options.solver;
verbose = options.verbose;
repeated = options.repeated;

if(isempty(solver))
    language = 'mosek';
end

if(isempty(verbose))
    verbose = 0;
end

if(isempty(repeated))
    repeated = 0;
end



weights = net.weights;
biases = net.biases;
activation = net.activation;
dims = net.dims;

if(~strcmp(activation,'relu'))
    error(['The method ReachSDP version', version ,'is currently supported for ReLU activation functions only.']);
end

% number of layers
%num_layers = numel(biases);

% input dimension
dim_in = dims(1);

% output dimension
%dim_out = dims(end);

dim_last_hidden = dims(end-1);

% total number of neurons
num_neurons = sum(dims(2:end-1));


%% outer approximation of the input set by a hyper-rectangle

dim_x = dim_in;
dim_u = size(Bsys,2);

Fx = input_polytope(:,1:end-1);
fx = input_polytope(:,end);

dim_px = length(fx);

x = sdpvar(dim_x,1);
x_min = zeros(dim_x,1);
x_max = zeros(dim_x,1);
options = sdpsettings('solver',solver,'verbose',0);
for i=1:dim_x
    optimize([Fx*x<=fx],x(i),options);
    x_min(i,1) = value(x(i));
    
    optimize([Fx*x<=fx],-x(i),options);
    x_max(i,1) = value(x(i));
end


% Interval arithmetic to find the activation bounds
[Y_min,Y_max] = net.interval_arithmetic(x_min,x_max);

X_min = net.activate(Y_min);
X_max = net.activate(Y_max);


Ip = find(Y_min>0);
In = find(Y_max<0);
Ipn = setdiff(1:num_neurons,union(Ip,In));

AA = ([blkdiag(weights{1:end-1}) zeros(num_neurons,dim_last_hidden)]);
BB = ([zeros(num_neurons,dim_in) eye(num_neurons)]);
bb = cat(1,biases{1:end-1});

% Entry selector matrices
E0 = [eye(dim_in) zeros(dim_in,num_neurons)];
El = [zeros(dim_last_hidden,num_neurons+dim_in-dim_last_hidden) eye(dim_last_hidden)];


%% Construct Min
tau = sdpvar(dim_px,dim_px,'symmetric');
constraints = [tau(:)>=0, diag(tau)==0];

CM_in = ([eye(dim_in) zeros(dim_in,num_neurons+1);zeros(1,dim_in+num_neurons) 1]);

P = [Fx'*tau*Fx -Fx'*tau*fx;-fx'*tau*Fx fx'*tau*fx];
Min = CM_in.'*P*CM_in;

%% Construct Mmid

T = zeros(num_neurons);
if(repeated)
    II = eye(num_neurons);
    C = [];
    if(numel(Ip)>1)
        C = nchoosek(Ip,2);
    end
    if(numel(In)>1)
        C = [C;nchoosek(In,2)];
    end
    C = nchoosek(1:num_neurons,2);
    m = size(C,1);
    if(m>0)
        if(strcmp(language,'cvx'))
            variable zeta(m,1) nonnegative;
        elseif(strcmp(language,'yalmip'))
            zeta = sdpvar(m,1);
            constraints = [constraints,zeta>=0];
        else
        end
        E = II(:,C(:,1))-II(:,C(:,2));
        T = E*diag(zeta)*E';
    end
end

nu = sdpvar(num_neurons,1);

lambda = sdpvar(num_neurons,1);

eta = sdpvar(num_neurons,1);

D = diag(sdpvar(num_neurons,1));

constraints = [constraints, nu(In)>=0, nu(Ipn)>=0, eta(Ip)>=0, eta(Ipn)>=0, D(:)>=0];

%

alpha_param = zeros(num_neurons,1);
alpha_param(Ip)=1;

beta_param = ones(num_neurons,1);
beta_param(In) = 0;

Q11 = -2*diag(alpha_param.*beta_param.*lambda);
Q12 = diag((alpha_param+beta_param).*lambda)+T;
Q13 = -nu;
Q22 = -2*diag(lambda)-2*D-2*T;
Q23 = nu+eta+D*(X_min+X_max);
Q33 = -2*X_min'*D*X_max;


Q = [Q11 Q12 Q13; Q12.' Q22 Q23;Q13.' Q23.' Q33];

CM_mid = [AA bb;BB zeros(size(BB,1),1);zeros(1,size(BB,2)) 1];

Mmid = CM_mid.'*Q*CM_mid;

%% Construct Mout

CM_out = [E0 zeros(size(E0,1),1); weights{end}*El biases{end};zeros(1,size(E0,2)) 1];

b = sdpvar(1);
obj = b;


bound = nan(size(c,2),1);
for i=1:size(c,2)
    
    cc = c(:,i);
    
    S = [zeros(dim_x) zeros(dim_x,dim_u) Asys'*cc;zeros(dim_u,dim_x) zeros(dim_u) Bsys'*cc;cc'*Asys cc'*Bsys -2*b];
    
    Mout = CM_out.'*S*CM_out;
    
    
    %% solve SDP
    
    options = sdpsettings('solver',solver,'verbose',verbose);
    out = optimize([constraints, Min+Mmid+Mout<=0],obj,options);
    
    bound(i,1) = value(obj);
    time(i,1)= out.solvertime;
    status{i} = out.info;
    
    message = ['method: ReachSDP ', version,'| solver: ', solver, '| bound: ', num2str(bound(i,1),'%.3f'), '| solvetime: ', num2str(time(i,1),'%.3f'), '| status: ', status{i}];
    
    disp(message);
    
end

end