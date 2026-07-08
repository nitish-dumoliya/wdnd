#*******************************************SETS******************************************#
set nodes;			   # Set of nodes/vertices
set pipes;			   # Set of commercial pipes available
set arcs within {i in nodes, j in nodes: i != j};	# Set of arcs/links/edges
set Source;		       # Set of source nodes 

set parallel_arcs within {i in nodes, j in nodes: i != j}:= {(46, 30), (46, 13), (48, 24), (48, 52)}; 
#****************************************PARAMETERS***************************************#
param L{arcs };		   # Total length of each arc/link
param E{nodes};		   # Elevation of each node
param P{nodes};		   # Minimum pressure required at each node
param pmax{nodes};		   # Maximum pressure required at each node
param D{nodes};		   # Demand of each node
param d{pipes};		   # Diameter of each commercial pipe
param C{pipes};		   # Cost per unit length of each commercial pipe
param R{arcs};		   # Roughness of each commercial pipe
param omega := 10.67;  # SI Unit Constant for Hazen Williams Equation
param vmax{arcs} default (sum {k in nodes diff Source} D[k])/((3.14/4)*(d[1])^2);
param p:= 1.852;

param alpha{k in pipes} :=
    omega / (130^1.852 * d[k]^4.87);

param Q_max = sum{k in nodes diff Source} D[k];
param D_min = min{i in nodes diff Source} D[i];
param D_max = max{i in nodes diff Source} D[i];
param d_max ;
param d_min ;

param delta := 0.01;
param R_min = min{(i,j) in arcs} R[i,j];
param R_max = max{(i,j) in arcs} R[i,j];

param MaxK{(i,j) in arcs} := omega * L[i,j] / (R_min^1.852 * d_min^4.87);

param eps{(i,j) in arcs} := 0.0535*(1e-3/MaxK[i,j])^(0.54);

#****************************************VARIABLES****************************************#
var l{arcs diff parallel_arcs,pipes} >= 0 ;	# Length of each commercial pipe for each arc/link
var l1{parallel_arcs,pipes} >= 0 ;	# Length of each commercial pipe for each arc/link
var l2{parallel_arcs,pipes} >= 0 ;	# Length of each commercial pipe for each arc/link

param y{arcs diff parallel_arcs};
param y1{parallel_arcs};
param y2{parallel_arcs};
#****************************************OBJECTIVE****************************************#
# Total cost as a sum of "length of the commercial pipe * cost per unit length of the commercial pipe"
minimize total_cost : sum{(i,j) in arcs diff parallel_arcs} sum{k in pipes}l[i,j,k]*C[k] + sum{(i,j) in parallel_arcs} sum{k in pipes}(l1[i,j,k]+l2[i,j,k])*C[k];	

#****************************************CONSTRAINTS**************************************#
# Smooth-Approximation of Hazen-Williams Constraint
subject to con2{(i,j) in arcs diff parallel_arcs}:
    sum{k in pipes}(omega * l[i,j,k] / (R[i,j]^1.852 * d[k]^4.87)) = L[i,j] * y[i,j];

subject to con3{(i,j) in parallel_arcs}:
    sum{k in pipes}(omega * l1[i,j,k] / (R[i,j]^1.852 * d[k]^4.87)) = L[i,j] * y1[i,j];

subject to con4{(i,j) in parallel_arcs}:
    sum{k in pipes}(omega * l2[i,j,k] / (R[i,j]^1.852 * d[k]^4.87)) = L[i,j] * y2[i,j];

subject to con5{(i,j) in arcs diff parallel_arcs}: 
    sum{k in pipes} l[i,j,k] - L[i,j] = 0 
;

subject to con6{(i,j) in arcs diff parallel_arcs, k in pipes}: 
    l[i,j,k] - L[i,j] <= 0 
;

subject to con7{(i,j) in parallel_arcs}: 
    sum{k in pipes} l1[i,j,k] - L[i,j] = 0 
;

subject to con8{(i,j) in parallel_arcs, k in pipes}: 
    l1[i,j,k] - L[i,j] <= 0 
;

subject to con9{(i,j) in parallel_arcs}: 
    sum{k in pipes} l2[i,j,k] - L[i,j] = 0 
;

subject to con10{(i,j) in parallel_arcs, k in pipes}: 
    l2[i,j,k] - L[i,j] <= 0 
;

#subject to con14{(i,j) in arcs}: 
#   -Q_max <= q[i,j] <= Q_max
#;
#*******************************************************************************************#
