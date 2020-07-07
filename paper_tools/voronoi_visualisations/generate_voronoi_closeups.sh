#!/bin/bash

# generate low_conc image
voro++ -o -y 280 310 280 310 150 170 B1_v500_25s_particle_positions.txt  # compute voronoi tessellation
povray +W1920 +H1080 +A0.01 +Ovoronoi_viz_lowconc.png pov_params_lowconc.pov  # render in povray
convert -trim voronoi_viz_lowconc.png voronoi_viz_lowconc.png  # autocrop
rm B1_v500_25s_particle_positions.txt.vol B1_v500_25s_particle_positions.txt_v.pov B1_v500_25s_particle_positions.txt_p.pov  # cleanup

# generate high_conc image
voro++ -o -y 280 310 280 310 280 300 B1_v500_25s_particle_positions.txt  # compute voronoi tessellation
povray +W1920 +H1080 +A0.01 +Ovoronoi_viz_highconc.png pov_params_highconc.pov  # render in povray
convert -trim voronoi_viz_highconc.png voronoi_viz_highconc.png  # autocrop
rm B1_v500_25s_particle_positions.txt.vol B1_v500_25s_particle_positions.txt_v.pov B1_v500_25s_particle_positions.txt_p.pov  # cleanup

