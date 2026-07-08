#*******************************************SETS******************************************#
set nodes;			   # Set of nodes/vertices
set pipes;			   # Set of commercial pipes available
set arcs within {i in nodes, j in nodes: i != j};	# Set of arcs/links/edges
set Source;		       # Set of source nodes 

set fixed_arcs within {i in nodes, j in nodes: i != j}:= {(1,2), (2,6), (5,4), (3, 4), (2, 3), (6, 7), (6,9), (7,10), (9,10)}; 
set parallel_arcs within {i in nodes, j in nodes: i != j}:= {(1,2), (2,6), (5,4)};
set unfixed_arcs within {i in nodes, j in nodes: i != j}:= {(4,8), (7,8), (8,11), (10,11), (11,12)};

#****************************************PARAMETERS***************************************#
param L{arcs };		   # Total length of each arc/link
param E{nodes};		   # Elevation of each node
param P{nodes};		   # Minimum pressure required at each node
param pmax{nodes};		   # Maximum pressure required at each node
param D{nodes};		   # Demand of each node
param d{pipes};		   # Diameter of each commercial pipe
param C{pipes};		   # Cost per unit length of each commercial pipe
param C_clean{pipes};		   # Cost per unit length of each cleaned pipe
param R{pipes};		   # Roughness of each commercial pipe
param omega := 10.67;  # SI Unit Constant for Hazen Williams Equation
param vmax{arcs} default (sum {k in nodes diff Source} D[k])/((3.14/4)*(d[1])^2);
param delta := 0.1;
param exdiam{arcs};
param excost{arcs};
param exroughness{arcs};
param p:= 1.852;
param Q_max = sum{k in nodes diff Source} D[k];
param D_min = min{i in nodes diff Source} D[i];
param D_max = max{i in nodes diff Source} D[i];
param d_min = min{i in pipes} d[i];
param d_max = max{i in pipes} d[i];

param alpha{k in pipes} :=
    omega / (100^1.852 * d[k]^4.87);

param R_min = min{k in pipes} R[k];
param R_max = max{k in pipes} R[k];

param MaxK{(i,j) in arcs} := omega * L[i,j] / (R_min^1.852 * d_min^4.87);
#param eps{arcs};

param eps{(i,j) in arcs} := 0.0535*(1e-3/MaxK[i,j])^(0.54);
#****************************************VARIABLES****************************************#
var l{parallel_arcs union unfixed_arcs,pipes} >= 0 ;	# Length of each commercial pipe for each arc/link
param y{parallel_arcs union unfixed_arcs};
#****************************************OBJECTIVE****************************************#
# Total cost as a sum of "length of the commercial pipe * cost per unit length of the commercial pipe"
minimize total_cost : sum{(i,j) in parallel_arcs union unfixed_arcs} sum{k in pipes} l[i,j,k]*C[k];	

#****************************************CONSTRAINTS**************************************#
subject to con2{(i,j) in parallel_arcs union unfixed_arcs}: 
   sum{k in pipes}(omega * l[i,j,k]/(R[k]^1.852 * d[k]^4.87)) = L[i,j]*y[i,j]
;

subject to con3{(i,j) in parallel_arcs union unfixed_arcs}: 
    sum{k in pipes} l[i,j,k] = L[i,j]
;

subject to con4{(i,j) in parallel_arcs union unfixed_arcs , k in pipes}: 
    l[i,j,k] <= L[i,j]
;

#*******************************************************************************************#
