function [Y_min,Y_max,X_min,X_max,out_min,out_max] = interval_arithmatic(net,x_min,x_max)
% estimate the output set using interval arithmatic for initialization of
% Reach-SDP

l{1} = x_min;
u{1} = x_max;

Y_min = [];
Y_max = [];

%X_min = l{1};
%X_max = u{1};

X_min = [];
X_max = [];
num_layers = length(net.dims)-2;

for i=1:num_layers
    l{i+1} = max(net.weights{i},0)*l{i}+min(net.weights{i},0)*u{i}+net.biases{i}(:);
    Y_min = [Y_min;l{i+1}];
    if(strcmp(net.activation,'relu'))
        l{i+1} = max(l{i+1},0);
    elseif(strcmp(net.activation,'tanh'))
        l{i+1} = tanh(l{i+1});
    elseif(strcmp(net.activation,'sigmoid'))
        l{i+1} = 1./(1+exp(-l{i+1}));
    end
    X_min = [X_min;l{i+1}];
    u{i+1} = min(net.weights{i},0)*l{i}+max(net.weights{i},0)*u{i}+net.biases{i}(:);
    Y_max = [Y_max;u{i+1}];
    if(strcmp(net.activation,'relu'))
        u{i+1} = max(u{i+1},0);
    elseif(strcmp(net.activation,'tanh'))
        u{i+1} = tanh(u{i+1});
    elseif(strcmp(net.activation,'sigmoid'))
        u{i+1} = 1./(1+exp(-u{i+1}));
    end
    X_max = [X_max;u{i+1}];
end

i = num_layers + 1;
if(strcmp(net.activation,'relu'))
    out_min = max(net.weights{i},0)*l{i}+min(net.weights{i},0)*u{i}+net.biases{i}(:);
    out_max = min(net.weights{i},0)*l{i}+max(net.weights{i},0)*u{i}+net.biases{i}(:);
else
    out_min = max(net.weights{i},0)*l{i}+min(net.weights{i},0)*u{i}+net.biases{i}(:);
    out_max = min(net.weights{i},0)*l{i}+max(net.weights{i},0)*u{i}+net.biases{i}(:);
end

X_min = double(X_min);
X_max = double(X_max);
Y_min = double(Y_min);
Y_max = double(Y_max);
out_min = double(out_min);
out_max = double(out_max);


end

