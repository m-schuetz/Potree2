
















export let vsMesh = `
#version 450

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldViewProj;
} uniforms;

layout(location = 0) in vec3 position;
layout(location = 1) in ivec4 color;

layout(location = 0) out vec4 vColor;

void main() {
	vColor = vec4(
		float(color.x) / 256.0,
		float(color.y) / 256.0,
		float(color.z) / 256.0,
		1.0
	);

	gl_Position = uniforms.worldViewProj * vec4(position, 1.0);
}
`;

export let fsMesh = `
#version 450

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vColor;
	// outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
`;
