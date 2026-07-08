#*******************************************SETS******************************************#
set nodes;			   # Set of nodes/vertices
set pipes;			   # Set of commercial pipes available
set arcs within {i in nodes, j in nodes: i != j};	# Set of arcs/links/edges
set Source;		       # Set of source nodes 
set fixarcs within {i in nodes, j in nodes: i != j};

#****************************************PARAMETERS***************************************#
param L{arcs };		   # Total length of each arc/link
param fix_L{fixarcs };		   # Total length of each arc/link
param E{nodes};		   # Elevation of each node
param P{nodes};		   # Minimum pressure required at each node
param pmax{nodes};		   # Maximum pressure required at each node
param D{nodes};		   # Demand of each node
param d{pipes};		   # Diameter of each commercial pipe
param C{pipes};		   # Cost per unit length of each commercial pipe
param R{pipes};		   # Roughness of each commercial pipe
param omega := 10.67;  # SI Unit Constant for Hazen Williams Equation
param vmax{arcs} default (sum {k in nodes diff Source} D[k])/((3.14/4)*(d[1])^2);
param p:= 1.852;

param fixdiam{fixarcs};
param alpha{k in pipes} :=
    omega / (R[k]^1.852 * d[k]^4.87);

param Q_max = sum{k in nodes diff Source} D[k];
param D_min = min{i in nodes diff Source} D[i];
param D_max = max{i in nodes diff Source} D[i];
param d_max = max{i in pipes} d[i];
param d_min = min{i in pipes} d[i];

param delta := 0.01;
param a := (15/8)*delta**(p-1) + (1/8)*(p-1)*p*delta**(p-1) - (7/8)*p*delta**(p-1);
param b := - (5/4)*delta**(p-3) - (1/4)*(p-1)*p*delta**(p-3) - (5/4)*p*delta**(p-3);
param c := (3/8)*delta**(p-5) + (1/8)*(p-1)*p*delta**(p-5) - (3/8)*p*delta**(p-5);

param R_min = min{k in pipes} R[k];
param R_max = max{k in pipes} R[k];

param MaxK{(i,j) in arcs} := omega * L[i,j] / (R_min^1.852 * d_min^4.87);

param eps{(i,j) in arcs} := 0.0535*(1e-3/MaxK[i,j])^(0.54);  # Using relative error of approx 2
#param eps{(i,j) in arcs} := 0.0153*(1e-3/MaxK[i,j])^(0.54); # Using relative error of approx 1


#****************************************VARIABLES****************************************#
var l{arcs,pipes} >= 0 ;	# Length of each commercial pipe for each arc/link
var l1{fixarcs,pipes} >= 0 ;	# Length of each commercial pipe for each arc/link
#var l2{fixarcs,pipes} >= 0 ;	# Length of each commercial pipe for each arc/link
var q{arcs};	            # Flow variable
var q1{fixarcs};	            # Flow variable
var q2{fixarcs};	            # Flow variable
#var q{arcs}>=-Q_max,<=Q_max;	            # Flow variable
var h{nodes};	            # Head
#****************************************OBJECTIVE****************************************#
# Total cost as a sum of "length of the commercial pipe * cost per unit length of the commercial pipe"
minimize total_cost : sum{(i,j) in arcs} sum{k in pipes}l[i,j,k]*C[k] + sum{(i,j) in fixarcs} sum{k in pipes} l1[i,j,k]*C[k];	

#****************************************CONSTRAINTS**************************************#
subject to con1{j in nodes diff Source}:
    sum{i in nodes : (i,j) in arcs }q[i,j] -  sum{i in nodes : (j,i) in arcs}q[j,i] -  D[j] = 0
;

#subject to con2{(i,j) in arcs}: 
#     h[i] - h[j]  = q[i,j]*abs(q[i,j])^0.852 * sum{k in pipes} (omega * l[i,j,k] / ( (R[k]^1.852) * (d[k])^4.87));

# Smooth-Approximation of Hazen-Williams Constraint
subject to con2{(i,j) in arcs diff fixarcs}: 
    #h[i] - h[j]  -  q[i,j]*abs(q[i,j])^0.852 * sum{k in pipes}(omega * l[i,j,k] / (R[k]^1.852 * d[k]^4.87)) = 0;
    h[i] - h[j]  -  (q[i,j]^3 * (q[i,j]^2 + eps[i,j]^2)^0.426 / (q[i,j]^2 + 0.426*eps[i,j]^2)) * sum{k in pipes}(omega * l[i,j,k] / (R[k]^1.852 * d[k]^4.87)) = 0;
    #h[i] - h[j]  -  (q[i,j] * (q[i,j]^2 + eps[i,j]^2)^0.426) * sum{k in pipes}(omega * l[i,j,k] / (R[k]^1.852 * d[k]^4.87)) = 0;

subject to con2_{(i,j) in fixarcs}: 
    #h[i] - h[j]  -  q1[i,j]*abs(q1[i,j])^0.852 * sum{k in pipes}(omega * l[i,j,k] / (R[k]^1.852 * d[k]^4.87)) = 0;
    h[i] - h[j]  -  (q1[i,j]^3 * (q1[i,j]^2 + eps[i,j]^2)^0.426 / (q1[i,j]^2 + 0.426*eps[i,j]^2)) * sum{k in pipes}(omega * l[i,j,k] / (R[k]^1.852 * d[k]^4.87)) = 0;
    #h[i] - h[j]  -  (q1[i,j] * (q1[i,j]^2 + eps[i,j]^2)^0.426) * sum{k in pipes}(omega * l[i,j,k] / (R[k]^1.852 * d[k]^4.87)) = 0;


subject to con3{(i,j) in arcs}: 
    sum{k in pipes} l[i,j,k] - L[i,j] = 0 
;

subject to con4{(i,j) in arcs, k in pipes}: 
    l[i,j,k] - L[i,j] <= 0 
;

subject to con2__{(i,j) in fixarcs}: 
    #h[i] - h[j]  -  q2[i,j]*abs(q2[i,j])^0.852 * sum{k in pipes}(omega * l1[i,j,k] / (R[k]^1.852 * d[k]^4.87)) = 0;
    h[i] - h[j]  -  (q2[i,j]^3 * (q2[i,j]^2 + eps[i,j]^2)^0.426 / (q2[i,j]^2 + 0.426*eps[i,j]^2)) * sum{k in pipes}(omega * l1[i,j,k] / (R[k]^1.852 * d[k]^4.87)) = 0;
    #h[i] - h[j]  -  (q2[i,j] * (q2[i,j]^2 + eps[i,j]^2)^0.426) * sum{k in pipes}(omega * l1[i,j,k] / (R[k]^1.852 * d[k]^4.87)) = 0;

subject to con3_{(i,j) in fixarcs}: 
    sum{k in pipes} l1[i,j,k] - fix_L[i,j] = 0 
;

subject to con4_{(i,j) in fixarcs , k in pipes}: 
    l1[i,j,k] - fix_L[i,j] <= 0 
;

subject to con5{i in Source}: 
    h[i] - E[i] = 0
;
subject to con6{i in nodes diff Source}: -h[i] + E[i] + P[i] <= 0;

subject to con8{(i,j) in fixarcs}:
    q[i,j] = q1[i,j] + q2[i,j];

#subject to con7{(i,j) in arcs}: 
#   -Q_max <= q[i,j] <= Q_max
#;
#*******************************************************************************************#
