#*******************************************SETS******************************************#
set nodes;			   # Set of nodes/vertices
set pipes;			   # Set of commercial pipes available
set arcs within {i in nodes, j in nodes: i != j};	# Set of arcs/links/edges
set Source;		       # Set of source nodes 
set fixarcs within {i in nodes, j in nodes: i != j};

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

param R_min = 100;
param R_max = 100;

param MaxK{(i,j) in arcs} := omega * L[i,j] / (R_min^1.852 * d_min^4.87);

param eps{(i,j) in arcs} := 0.0535*(1e-3/MaxK[i,j])^(0.54);
 
#****************************************VARIABLES****************************************#
var l{unfixed_arcs, pipes}>=0;
var l1{parallel_arcs, pipes}>=0;
param y{unfixed_arcs union parallel_arcs};

#****************************************OBJECTIVE****************************************#
# Total cost as a sum of "length of the commercial pipe * cost per unit length of the commercial pipe"
minimize total_cost:sum{(i,j) in unfixed_arcs} sum{k in pipes}l[i,j,k]*C[k] + sum{(i,j) in parallel_arcs} sum{k in pipes}(l1[i,j,k])*C[k];
#****************************************CONSTRAINTS**************************************#
subject to con2{(i,j) in unfixed_arcs}:
   sum{k in pipes}(omega * l[i,j,k]/(R[k]^1.852 * d[k]^4.87)) = L[i,j]*y[i,j];

subject to con3{(i,j) in parallel_arcs}:
   sum{k in pipes}(omega * l1[i,j,k]/(R[k]^1.852 * d[k]^4.87)) = L[i,j]*y[i,j];

subject to con6{(i,j) in unfixed_arcs}: sum{k in pipes} l[i,j,k] = L[i,j];
subject to con7{(i,j) in unfixed_arcs, k in pipes}: l[i,j,k] <= L[i,j];

subject to con8{(i,j) in parallel_arcs}: sum{k in pipes} l1[i,j,k] = L[i,j];
subject to con9{(i,j) in parallel_arcs, k in pipes}: l1[i,j,k] <= L[i,j];

#subject to con15{(i,j) in arcs}: -Q_max <= q[i,j];
#
#subject to con16{(i,j) in arcs}: q[i,j] <= Q_max;
#*******************************************************************************************#
