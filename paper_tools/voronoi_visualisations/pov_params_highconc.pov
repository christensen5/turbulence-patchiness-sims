#version 3.6;

// Right-handed coordinate system in which the z-axis points upwards
camera {
	location <100, 0, 310>
	sky z
	right -0.24*x*image_width/image_height
	up 0.24*z
	look_at <295,295,290>
}

// White background
background{rgb 1}

// Two lights with slightly different colors
light_source{<-8,-20,30> color rgb <0.77,0.75,0.75>}
light_source{<25,-12,12> color rgb <0.38,0.40,0.40>}

// Radius of the Voronoi cell network
#declare r=0.08;

// Radius of the particles
#declare s=0.5;

// Particles
union{
#include "B1_v500_25s_particle_positions.txt_p.pov"
	pigment{rgb <1, 0, 0>} finish{reflection 0.1 specular 0.3 ambient 0.42}
}

// Voronoi cells
union{
#include "B1_v500_25s_particle_positions.txt_v.pov"
	pigment{rgb <0.2,0.2,0.2>} finish{specular 0.5 ambient 0.42}
}
