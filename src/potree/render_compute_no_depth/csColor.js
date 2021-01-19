
export let group_size = "128";

export let csColor = `

#version 450

layout(local_size_x = ${group_size}, local_size_y = 1) in;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldView;
	mat4 proj;
	uint width;
	uint height;
	uint numPoints;
} uniforms;


layout(std430, set = 0, binding = 1) buffer SSBO_COLORS {
	uint ssbo_colors[];
};

layout(std430, set = 0, binding = 2) buffer SSBO_position {
	float positions[];
};

layout(std430, set = 0, binding = 3) buffer SSBO_color {
	uint colors[];
};


void main(){

	uint index = gl_GlobalInvocationID.x;

	vec4 pos_point = vec4(
		positions[3 * index + 0],
		positions[3 * index + 1],
		positions[3 * index + 2],
		1.0);

	vec4 viewPos = uniforms.worldView * pos_point;
	vec4 pos = uniforms.proj * viewPos;

	pos.xyz = pos.xyz / pos.w;

	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		return;
	}

	ivec2 imageSize = ivec2(
		int(uniforms.width),
		int(uniforms.height)
	);

	vec2 imgPos = (pos.xy * 0.5 + 0.5) * imageSize;
	ivec2 pixelCoords = ivec2(imgPos);
	int pixelID = pixelCoords.x + pixelCoords.y * imageSize.x;

	uint color = colors[index];

	ssbo_colors[pixelID] = color;
}

`;

