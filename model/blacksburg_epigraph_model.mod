#*******************************************SETS******************************************#
set nodes;			   # Set of nodes/vertices
set pipes;			   # Set of commercial pipes available
set arcs within {i in nodes, j in nodes: i != j};	# Set of arcs/links/edges
set Source;		       # Set of source nodes 
set fixarcs within {i in nodes, j in nodes: i != j};

param NP integer > 1 := card(pipes);
set segs := 1..NP-1;


#****************************************PARAMETERS***************************************#
param L{arcs };		   # Total length of each arc/link
param E{nodes};		   # Elevation of each node
param P{nodes};		   # Minimum pressure required at each node
param pmax{nodes};		   # Maximum pressure required at each node
param D{nodes};		   # Demand of each node
param d{pipes};		   # Diameter of each commercial pipe
param C{pipes};		   # Cost per unit length of each commercial pipe
param R{pipes};		   # Roughness of each commercial pipe
param omega := 10.67;  # SI Unit Constant for Hazen Williams Equation
param vmax{arcs} default (sum {k in nodes diff Source} D[k])/((3.14/4)*(d[1])^2);
param delta := 0.1;
param fixdiam{fixarcs};
param p:= 1.852;
param Q_max = sum{k in nodes diff Source} D[k];
param D_min = min{i in nodes diff Source} D[i];
param D_max = max{i in nodes diff Source} D[i];
param d_min;
param d_max;
param fix_r{fixarcs};
param fix_c{fixarcs};

param alpha{k in pipes} :=
    omega / (R[k]^1.852 * d[k]^4.87);

param alpha_min := min{k in pipes} alpha[k];
param alpha_max := max{k in pipes} alpha[k];


param R_min = min{k in pipes} R[k];
param R_max = max{k in pipes} R[k];

param MaxK{(i,j) in arcs} := omega * L[i,j] / (R_min^1.852 * d_min^4.87);

param eps{(i,j) in arcs} := 0.0535*(1e-3/MaxK[i,j])^(0.54);

# Segment slope
param slope{s in segs} :=
    ( C[s] - C[s+1] ) /
    ( alpha[s] - alpha[s+1] );

# Segment intercept
param intercept{s in segs} :=
    (alpha[s]*C[s+1] - alpha[s+1] * C[s]) / ( alpha[s] - alpha[s+1] );

#****************************************VARIABLES****************************************#
var q{arcs};	            # Flow variable
var h{nodes};	            # Head
var y{arcs}>=alpha_min,<=alpha_max;
var z{arcs};
#****************************************OBJECTIVE****************************************#
minimize total_cost:sum{(i,j) in arcs diff fixarcs} z[i,j] + sum{(i,j) in fixarcs} L[i,j]*fix_c[i,j] ;
#minimize total_cost:sum{(i,j) in arcs diff fixarcs} z[i,j] ;

#****************************************CONSTRAINTS**************************************#
subject to con1{j in nodes diff Source}:
    sum{i in nodes : (i,j) in arcs }q[i,j] -  sum{i in nodes : (j,i) in arcs}q[j,i] =  D[j];

subject to con2{(i,j) in arcs diff fixarcs}: 
#   h[i] - h[j]  = (q[i,j]*abs(q[i,j])^0.852) * L[i,j] * y[i,j];
   h[i] - h[j]  = (q[i,j])^3 *((((q[i,j])^2 + eps[i,j]^2)^0.426) /((q[i,j])^2 + 0.426*eps[i,j]^2)) * L[i,j] * y[i,j];
#   h[i] - h[j]  = q[i,j] *(((q[i,j])^2 + eps[i,j]^2)^0.426) * sum{k in pipes}(omega * l[i,j,k]/(R[k]^1.852 * d[k]^4.87));

subject to con2_{(i,j) in fixarcs}:
#   h[i] - h[j]  = (q[i,j]*abs(q[i,j])^0.852) * omega * L[i,j] / (fix_r[i,j]^1.852 * fixdiam[i,j]^4.87);
   h[i] - h[j]  = (q[i,j])^3 *((((q[i,j])^2 + eps[i,j]^2)^0.426) /((q[i,j])^2 + 0.426*eps[i,j]^2)) * omega * L[i,j] / (fix_r[i,j]^1.852 * fixdiam[i,j]^4.87);
#   h[i] - h[j]  = q[i,j] *(((q[i,j])^2 + eps[i,j]^2)^0.426) * omega * L[i,j] / (fix_r[i,j]^1.852 * fixdiam[i,j]^4.87);

subject to con5{i in Source}: h[i] = E[i];

subject to con6{i in nodes diff Source}: h[i] <= E[i] + pmax[i];

subject to con7{i in nodes diff Source}: E[i] + P[i] <= h[i];

subject to exact_cost{(i,j) in arcs diff fixarcs, s in segs}:
    z[i,j] >= L[i,j]*(slope[s] * y[i,j] + intercept[s]);

#subject to con8{(i,j) in arcs}: 
#   -Q_max <= q[i,j] <= Q_max
#;

#*******************************************************************************************#
