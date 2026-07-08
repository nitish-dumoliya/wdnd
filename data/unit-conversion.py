# Conversion factor from GPM to m^3/s
GPM_TO_CUBIC_MPS = 0.0000630902
FT_TO_METER = 0.3048

# List of node demands in GPM
node_demands_gpm = [
    1548.63, -52.11, -50.58, -25.77, -13.84, -53.65, -51.73,
    -200.58, -11.35, -10.77, -52.11, -100.77, -27.11, -100.77,
    -25.77, -51.54, -52.11, -10.38, -103.65, -52.11, -10.96,
    -51.35, -11.73, -51.54, -102.50, -51.54, -52.50, -50.96,
    -25.58, -41.92, -51.35
]

elevations_ft = [
    2163, 2141, 2132, 2121, 2153.5, 2141.5, 2129, 2127,
    2127, 2109.5, 2121, 2139, 2110, 2136.5, 2143, 2144.5,
    2149, 2109, 2144, 2149.5, 2140, 2141.5, 2144, 2156.5,
    2178, 2118, 2099.5, 2102, 2098.5, 2120, 2123
]

# Convert to m³/s
demands_m3s= [round(-gpm * GPM_TO_CUBIC_MPS, 8) for gpm in node_demands_gpm]

# Convert elevations to meters
elevations_m = [round(ft * FT_TO_METER, 6) for ft in elevations_ft]

# Print results

for i, (d, e) in enumerate(zip(demands_m3s, elevations_m)):
    print(f"{i}    {e}    {d} ")
