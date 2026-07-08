#*******************************************SETS******************************************#
set nodes;			   # Set of nodes/vertices
set pipes;			   # Set of commercial pipes available
set arcs within {i in nodes, j in nodes: i != j};	# Set of arcs/links/edges
set Source;		       # Set of source nodes 

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
param R{arcs};		   # Roughness of each commercial pipe
param omega := 10.67;  # SI Unit Constant for Hazen Williams Equation
param vmax{arcs} default (sum {k in nodes diff Source} D[k])/((3.14/4)*(d[1])^2);
param delta := 0.1;
param exdiam{arcs};
param excost{arcs};
param p:= 1.852;
param Q_max = sum{k in nodes diff Source} D[k];
param D_min = min{i in nodes diff Source} D[i];
param D_max = max{i in nodes diff Source} D[i];
param d_min = min{i in pipes} d[i];
param d_max = max{i in pipes} d[i];

param alpha{k in pipes} :=
    omega / (100^1.852 * d[k]^4.87);


param R_min = min{(i,j) in arcs} R[i,j];
param R_max = max{(i,j) in arcs} R[i,j];

param MaxK{(i,j) in arcs} := omega * L[i,j] / (R_min^1.852 * d_min^4.87);
#param eps{arcs};

# Segment slope
param slope{s in segs} :=
    ( C[s] - C[s+1] ) /
    ( alpha[s] - alpha[s+1] );

# Segment intercept
param intercept{s in segs} :=
    (alpha[s]*C[s+1] - alpha[s+1] * C[s]) / ( alpha[s] - alpha[s+1] );

param alpha_min := min{k in pipes} alpha[k];
param alpha_max := max{k in pipes} alpha[k];

param eps{(i,j) in arcs} := 0.0535*(1e-3/MaxK[i,j])^(0.54);

#****************************************VARIABLES****************************************#
var q{arcs};	            # Flow variable
var q1{arcs};	            # Flow variable
var q2{arcs};	            # Flow variable
var h{nodes};	            # Head
var y{arcs}>=alpha_min,<=alpha_max;
var z{arcs};

#****************************************OBJECTIVE****************************************#
minimize total_cost:sum{(i,j) in arcs} z[i,j];

#****************************************CONSTRAINTS**************************************#
subject to con1{j in nodes diff Source}:
    sum{i in nodes : (i,j) in arcs }(q1[i,j] + q2[i,j]) -  sum{i in nodes : (j,i) in arcs}(q1[j,i] + q2[j,i]) =  D[j]
;

subject to con2{(i,j) in arcs}: 
#   h[i] - h[j]  = (q1[i,j]*abs(q1[i,j])^0.852) * omega * L[i,j] / ( (R[i,j]^1.852) * (exdiam[i,j])^4.87);
   h[i] - h[j]  = (q1[i,j])^3 *((((q1[i,j])^2 + eps[i,j]^2)^0.426) /((q1[i,j])^2 + 0.426*eps[i,j]^2)) * omega * L[i,j] / ( (R[i,j]^1.852) * (exdiam[i,j])^4.87);

subject to con3{(i,j) in arcs}: 
#   h[i] - h[j]  = (q2[i,j]*abs(q2[i,j])^0.852) * L[i,j] * y[i,j] ;
   h[i] - h[j]  = (q2[i,j])^3 *((((q2[i,j])^2 + eps[i,j]^2)^0.426) /((q2[i,j])^2 + 0.426*eps[i,j]^2)) * L[i,j] * y[i,j] ;

subject to con6{i in Source}: 
    h[i] = E[i]
;

subject to con7{i in nodes diff Source}: h[i] >= (E[i] + P[i]) ;

subject to exact_cost{(i,j) in arcs, s in segs}:
    z[i,j] >= L[i,j]*(slope[s] * y[i,j] + intercept[s]);

subject to con9{(i,j) in arcs}: q[i,j] = q1[i,j] + q2[i,j];

#subject to con8{(i,j) in arcs}: 
#   -Q_max <= q[i,j] <= Q_max
#;
#*******************************************************************************************#
