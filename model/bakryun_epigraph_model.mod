#*******************************************SETS******************************************#
set nodes;			   # Set of nodes/vertices
set pipes;			   # Set of commercial pipes available
set arcs within {i in nodes, j in nodes: i != j};	# Set of arcs/links/edges
set Source;		       # Set of source nodes 
set fixarcs within {i in nodes, j in nodes: i != j};

param NP integer > 1 := card(pipes);
set segs := 1..NP-1;

set unfixed_arcs within {i in nodes, j in nodes: i != j}:= {(99, 1), (1, 2), (2, 3)}; 
set parallel_arcs within {i in nodes, j in nodes: i != j}:= {(3, 4), (4, 6), (6, 7), (7, 11), (11, 16), (16, 17)};
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
param d_min := min{k in pipes} d[k];
param d_max := max{k in pipes} d[k];
param fix_r{fixarcs};
param fix_c{fixarcs};

param alpha{k in pipes} :=
    omega / (R[k]^1.852 * d[k]^4.87);

param alpha_min := min{k in pipes} alpha[k];
param alpha_max := max{k in pipes} alpha[k];

param R_min = 100;
param R_max = 100;

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
var y{unfixed_arcs union parallel_arcs}>=alpha_min,<=alpha_max;
var z{unfixed_arcs union parallel_arcs};

var q{arcs};	            # Flow variable
var q1{parallel_arcs};	            # Flow variable
var q2{parallel_arcs};	            # Flow variable
var h{nodes};	            # Head
#****************************************OBJECTIVE****************************************#
# Total cost as a sum of "length of the commercial pipe * cost per unit length of the commercial pipe"
#minimize total_cost:sum{(i,j) in unfixed_arcs} sum{k in pipes}l[i,j,k]*C[k] + sum{(i,j) in parallel_arcs} sum{k in pipes}(l1[i,j,k])*C[k]; 
minimize total_cost:sum{(i,j) in unfixed_arcs} z[i,j] + sum{(i,j) in parallel_arcs} z[i,j]; 
#****************************************CONSTRAINTS**************************************#
subject to con1{j in nodes diff Source}:
    sum{i in nodes : (i,j) in arcs }q[i,j] -  sum{i in nodes : (j,i) in arcs}q[j,i] =  D[j];

#subject to con2{(i,j) in unfixed_arcs}:
#   h[i] - h[j]  = q[i,j]*abs(q[i,j])^0.852 * L[i,j] * y[i,j];
#
#subject to con3{(i,j) in parallel_arcs}:
#   h[i] - h[j]  = q1[i,j]*abs(q1[i,j])^0.852 * L[i,j] * y[i,j];
#
#subject to con4{(i,j) in parallel_arcs}: 
#   h[i] - h[j]  = q2[i,j]*abs(q2[i,j])^0.852 * omega * L[i,j] / (fix_r[i,j]^1.852 * fixdiam[i,j]^4.87);
# 
#subject to con5{(i,j) in fixarcs diff parallel_arcs}:
#   h[i] - h[j]  = q[i,j]*abs(q[i,j])^0.852 * omega * L[i,j] / (fix_r[i,j]^1.852 * fixdiam[i,j]^4.87);


subject to con2{(i,j) in unfixed_arcs}:
   h[i] - h[j]  = (q[i,j])^3 *((((q[i,j])^2 + eps[i,j]^2)^0.426) /((q[i,j])^2 + 0.426*eps[i,j]^2)) * L[i,j] * y[i,j];

subject to con3{(i,j) in parallel_arcs}:
   h[i] - h[j]  = (q1[i,j])^3 *((((q1[i,j])^2 + eps[i,j]^2)^0.426) /((q1[i,j])^2 + 0.426*eps[i,j]^2)) * L[i,j] * y[i,j];

subject to con4{(i,j) in parallel_arcs}: 
   h[i] - h[j]  = (q2[i,j])^3 *((((q2[i,j])^2 + eps[i,j]^2)^0.426) /((q2[i,j])^2 + 0.426*eps[i,j]^2)) * omega * L[i,j] / (fix_r[i,j]^1.852 * fixdiam[i,j]^4.87);
 
subject to con5{(i,j) in fixarcs diff parallel_arcs}:
   h[i] - h[j]  = (q[i,j])^3 *((((q[i,j])^2 + eps[i,j]^2)^0.426) /((q[i,j])^2 + 0.426*eps[i,j]^2)) * omega * L[i,j] / (fix_r[i,j]^1.852 * fixdiam[i,j]^4.87);

subject to exact_cost{(i,j) in unfixed_arcs union parallel_arcs, s in segs}: z[i,j] >= L[i,j]*(slope[s]*y[i,j] + intercept[s]);

subject to con12{i in Source}: h[i] = E[i];

subject to con13{i in nodes diff Source}: E[i] + P[i] <= h[i];

subject to con14{(i,j) in parallel_arcs}: q[i,j] = q1[i,j] + q2[i,j];

#subject to con15{(i,j) in arcs}: 
#   -Q_max <= q[i,j] <= Q_max
#;

#*******************************************************************************************#
