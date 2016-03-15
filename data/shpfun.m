clear all ; format long;
clc;


%% Lagrange shape functions in the interval [0,1]

xi = 0.211325;
eta = 0.211325;

L = 1; % [0,1] interval

v1 = [(1-xi)/ L, (xi/L) ];
v2 = [(1-eta)/L, (eta/L)]';
k = 1;

for j1 = 1:2
    for j2 = 1:2
    N_Lagrange(k) = v1(j1) * v2(j2);
    k = k+1;
    end
end


N_Lagrange;

clc;

%% Legendre shape functions in the interval [0,1]


xivec = [0.5];
etavec = [0 ];

for int1 = 1:length(xivec)
    xi = xivec(int1);
    for int2 = 1:length(etavec)
        eta = etavec(int2);

v2_legendre = [1, 2*xi - 1, 6*xi^2 - 6*xi + 1, 20*xi^3 - 30*xi^2 + 12*xi -1];
v1_legendre = [1, 2*eta - 1, 6*eta^2 - 6*eta + 1, 20*eta^3 - 30*eta^2 + 12*eta -1];

v1_legendre_ortho = [1, ...
    (2*xi-1) / (1/sqrt(3)),...
    (6*xi^2 - 6*xi + 1) / (sqrt(0.2)),...
    (20*xi^3 - 30*xi^2 + 12*xi - 1) / (sqrt (0.14285))];


v2_legendre_ortho = [1, (2*eta-1) / (1/sqrt(3)), ...
    (6*eta^2 - 6*eta + 1) / (sqrt(0.2)), ...
    (20*eta^3 - 30*eta^2 + 12*eta - 1) / (sqrt (0.14285)) ];


k = 1;

for j1 = 1:2
    for j2 = 1:2
    N_Legendre(k) =v1_legendre (j1) * v2_legendre(j2);
    N_Legendre_ortho (k) = v1_legendre_ortho (j1) * v2_legendre_ortho (j2) ;
    k = k+1;
    end
end
[xi,eta]
display(N_Legendre_ortho,'phi:') ;


    
    end
end