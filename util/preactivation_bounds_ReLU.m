function [l,u] = preactivation_bounds_ReLU(W,b,x_min,x_max)

X = 0.5*(x_min+x_max);
epsilon = 0.5*(x_max-x_min);


num_layers = numel(W)-1;
l=[];
u=[];
[ll,uu]=first_pre_activate(W{1},b{1},X,epsilon);
l=[l;ll];
u=[u;uu];
for j=1:num_layers-1
    [ll,uu]=next_layer_prebound(W{j+1},b{j+1},ll,uu);
    l=[l;ll];
    u=[u;uu];
end

    function  [l_first, u_first] = first_pre_activate(W,b,X,epsilon)
        l_first=-abs(W)*epsilon  +  W*X  +  b;
        u_first= abs(W)*epsilon  +  W*X  +  b;
    end

    function   [l_next, u_next] = next_layer_prebound(W,b,l_pre,u_pre)
        leng=length(l_pre);
        alph_upp=[];
        alph_low=[];
        alphabet_upp=[];
        alphabet_low=[];
        for i=1:leng
            if l_pre(i,1)>0
                alph_upp=[alph_upp;   1];
                alphabet_upp=[alphabet_upp;  0];
                alph_low=[ alph_low ; 1 ];
                alphabet_low=[ alphabet_low ; 0  ];
            elseif u_pre(i,1)<0
                alph_upp=[alph_upp ; 0 ];
                alphabet_upp=[alphabet_upp ; 0   ];
                alph_low=[alph_low ; 0 ];
                alphabet_low=[alphabet_low ; 0 ];
            else
                
                alph_upp=[alph_upp  ;  u_pre(i,1)/(u_pre(i,1)-l_pre(i,1))   ];
                alphabet_upp=[alphabet_upp  ; -l_pre(i,1)*u_pre(i,1)/(u_pre(i,1)-l_pre(i,1))  ];
                
                if u_pre(i,1)>abs(l_pre(i,1))
                    alph_low=[alph_low ; 1  ];
                    alphabet_low=[alphabet_low ; 0 ];
                else
                    alph_low=[alph_low ; 0];
                    alphabet_low=[alphabet_low ; 0 ];
                end
            end
        end
        
        W_pos=0.5*(W+abs(W));
        W_neg=0.5*(W-abs(W));
        mid=0.5*(l_pre+u_pre);
        dif=0.5*(u_pre-l_pre);
        cc1=  W_pos.*alph_low.'   +    W_neg.*alph_upp.';
        dd1=  W_pos*alphabet_low  +    W_neg*alphabet_upp + b;
        l_next=-abs(cc1)*dif   + cc1*mid + dd1 ;
    
    
    
        cc2=  W_neg.*alph_low.'    +    W_pos.*alph_upp.';
        dd2=  W_neg*alphabet_low   +    W_pos*alphabet_upp + b;
        u_next= abs(cc2)*dif   +  cc2*mid + dd2;
        
    end
end