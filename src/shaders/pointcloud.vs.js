
export let vs = `

#version 450

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldViewProj;
	ivec4 imin;
	vec4 offset;
} uniforms;

layout(location = 0) in ivec3 a_position;
layout(location = 1) in ivec4 a_rgb;
//<!-- POINT ATTRIBUTES -->

layout(location = 0) out vec4 vColor;

void main() {
	vColor = vec4(
		float(a_rgb.x) / 256.0,
		float(a_rgb.y) / 256.0,
		float(a_rgb.z) / 256.0,
		1.0
	);

	ivec3 min = uniforms.imin.xyz;

	int ix = (a_position.x) / 1000;
	int iy = (a_position.y) / 1000;
	int iz = (a_position.z) / 1000;
	
	ix = ix / 1000;
	iy = iy / 1000;
	iz = iz / 1000;

	vec3 pos = vec3(
		float(ix) * 0.0031996278762817386,
		float(iy) * 0.004269749641418458,
		float(iz) * 0.004647889137268066
	);

	pos = pos + uniforms.offset.xyz;

	// pos = pos + vec3(
	// 	-0.748212993144989,
	// 	-2.7804059982299805,
	// 	2.547821283340454);

	gl_Position = uniforms.worldViewProj * vec4(pos, 1.0);
}


`;