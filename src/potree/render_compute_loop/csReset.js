
export let csReset = `

#version 450

layout(local_size_x = 128, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer SSBO {
	uint framebuffer[];
};

layout(set = 0, binding = 1) uniform Uniforms {
	uint value;
} uniforms;

void main(){

	uint index = gl_GlobalInvocationID.x;

	framebuffer[index] = uniforms.value;
}
`;