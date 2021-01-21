
export let group_size = "128";

export let csPrepare = `

#version 450

struct Vertex{
	float x;
	float y;
	float z;
	uint color;
	uint pointID;
};

layout(local_size_x = ${group_size}, local_size_y = 1) in;

layout(set = 0, binding = 0) uniform Uniforms {
	uint width;
	uint height;
} uniforms;

layout(std430, set = 0, binding = 1) buffer SSBO_meta {
	uint numPoints;
	uint indirect_x;
	uint indirect_y;
	uint indirect_z;
} meta;

layout(std430, set = 0, binding = 2) buffer SSBO_colors {
	uint ssbo_colors[];
};

// layout(std430, set = 0, binding = 3) buffer SSBO_prep_pos {
// 	float ssbo_prep_pos[];
// };

// layout(std430, set = 0, binding = 4) buffer SSBO_prep_col {
// 	uint ssbo_prep_col[];
// };

layout(std430, set = 0, binding = 3) buffer SSBO_prep {
	Vertex ssbo_prep[];
};


layout(std430, set = 0, binding = 5) buffer SSBO_point_id {
	uint ssbo_point_id[];
};

layout(std430, set = 0, binding = 6) buffer SSBO_position {
	float ssbo_position[];
};



void main(){

	uint pixelID = gl_GlobalInvocationID.x;
	uint pointID = ssbo_point_id[pixelID];

	if(pointID == 0){
		return;
	}

	uint prepID = atomicAdd(meta.numPoints, 1);
	
	Vertex v;
	v.x = ssbo_position[3 * pointID + 0];
	v.y = ssbo_position[3 * pointID + 1];
	v.z = ssbo_position[3 * pointID + 2];
	// v.color = ssbo_colors[pixelID];
	v.pointID = pointID;

	uint R = ssbo_colors[4 * pixelID + 0];
	uint G = ssbo_colors[4 * pixelID + 1];
	uint B = ssbo_colors[4 * pixelID + 2];
	uint C = ssbo_colors[4 * pixelID + 3];

	uint r = R / C;
	uint g = G / C;
	uint b = B / C;

	uint color = 0 
		| (r & 0xFF) << 0
		| (g & 0xFF) << 8
		| (b & 0xFF) << 16;

	v.color = color;



	ssbo_prep[prepID] = v;

	// prepare indirect draw call buffer
	uint groups = prepID / 128;
	atomicMax(meta.indirect_x, groups);
	meta.indirect_y = 1;
	meta.indirect_z = 1;
}

`;
