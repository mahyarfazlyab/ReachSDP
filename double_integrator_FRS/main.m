%% Reach-SDP with Forward Reachability
clear all; clc
close all;
addpath('./util', './networks')


%% System Parameters
% double integrator system (ts = 1.0)
A = [1 1; 0 1];
B = [0.5; 1];
sys.uub =  1;
sys.ulb = -1;
load nnmpc_nets_di_1

C = eye(2);
n = size(B,1);
m = size(B,2);
sys.A = A;
sys.B = B;

% get network parameters
net = convert_nnmpc_to_net(weights, biases, 'relu', []);


%% Setup
% initial set
X0_b = [2.0; -1.0; 2.5; -1.5];
X0_poly = Polyhedron([1 0; -1 0; 0 1; 0 -1], X0_b);
X0 = X0_poly.outerApprox; % normalize the A matrix
X0_vec = X0;

dx = 0.02; % shrink the tube for better visualization
X0_poly_s = Polyhedron([1 0; -1 0; 0 1; 0 -1], X0_b-dx);

% SDP parameters
repeated = true;
verbose = false;

% reachability horizon
N = 6;

% facets of the output polytope
% A_out = [1 0; -1 0; 0 1; 0 -1; -1 1; 1 -1; 1 1; -1 -1];
% A_out = [1 0; -1 0; 0 1; 0 -1; -1 1; 1 -1; 1 1; -1 -1; 1 2; -1 -2; -1 2; 1 -2; 2 1; -2 -1; 2 -1; -2 1];
A_out = [1 0; -1 0; 0 1; 0 -1; -1 1; 1 -1; 1 1; -1 -1; 1 2; -1 -2; 1 4; -1 -4; -1 2; 1 -2; -1 4; 1 -4];

disp(['Starting FRS computation, N = ', num2str(N)]);


%% Gridding to Compute the Exact Reachable Sets
Xg_cell = {}; % grid-based reachable sets
Ug_cell = {}; % grid-based control sets
Xg = grid(X0_poly_s,40);
Xg_cell{end+1} = Xg;
for k = 1:N
    Xg_k = []; % one-step FRS at time k
    Ug_k = []; % one-step control set at time k
    for x = Xg_cell{end}'
        u = fwd_prop(net,x);
        x_next = A*x + B*proj(u(1),sys.ulb,sys.uub);
        Xg_k = [Xg_k; x_next'];
        Ug_k = [Ug_k; u(1)];
    end
    Xg_cell{end+1} = Xg_k;
    Ug_cell{end+1} = Ug_k;
end


%% Reach-SDP
poly_cell = cell(1,N+1);
poly_cell{1,1}  = X0_vec;

for i = 1:length(X0_vec)
    
    % polytopic initial set
    input_set.type = 'polytope';
    input_set.set  = X0_vec(i);
    poly_seq_vec   = X0_vec(i);
    
    % forward reachability
    for k = 1:N
        b_out = [];
        for c = A_out'
            sol   = reach_sdp_poly(net, input_set, c, repeated, sys, verbose);
            b_out = [b_out; sol.bound];
        end
        
        % shift horizon
        input_set.set = Polyhedron(A_out, b_out);
        
        % save results
        poly_seq_vec = [poly_seq_vec Polyhedron(A_out, b_out)];
        
        % report
        disp(['Reach-SDP Progress: N = ', num2str(k), ', i = ',...
            num2str(i), ', volume: ', num2str(input_set.set.volume)]);
    end
    poly_cell{1,i} = poly_seq_vec;
end


%% Plot results
figure('Renderer', 'painters')
hold on
% initial set
plot(X0_poly,'color','k','alpha',0.1)

% N-step FRS
for i = 1:length(X0_vec)
    for k = 1:N+1
        FRS_V = poly_cell{1,i}(k).V;
        FRS_V_bd = FRS_V(boundary(FRS_V(:,1), FRS_V(:,2), 0.0),:);
        if k == 1
            continue
%             plot(FRS_V_bd(:,1),FRS_V_bd(:,2),'k-','LineWidth',2)
        else
            plot(FRS_V_bd(:,1),FRS_V_bd(:,2),'r-','LineWidth',3)
        end
    end
end

% gridding states
for k = 2:N+1
    FRS = Xg_cell{k};
    FRS_bd = FRS(boundary(FRS(:,1), FRS(:,2), 0.5),:);
    plot(FRS_bd(:,1),FRS_bd(:,2),'b-','LineWidth',1.5)
end

grid off
axis equal
xlim([-1,6])
ylim([-3,3])
xlabel('$x_1$','Interpreter','latex')
ylabel('$x_2$','Interpreter','latex')


%% Auxiliary Functions
function out = proj(u, u_min, u_max)
    out = min(max(u,u_min),u_max);
end