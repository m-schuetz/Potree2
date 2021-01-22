
export let group_size = "128";
let depthPrecision = "1000.0";

export let csReproject = `

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
	mat4 worldView;
	mat4 proj;
	uint width;
	uint height;
	uint firstPoint;
	uint numPoints;
} uniforms;


layout(std430, set = 0, binding = 1) buffer SSBO_depth {
	uint ssbo_depth[];
};

layout(std430, set = 0, binding = 2) buffer SSBO_color {
	uint ssbo_color[];
};

layout(std430, set = 0, binding = 3) buffer SSBO_point_id {
	uint ssbo_point_id[];
};

layout(std430, set = 0, binding = 4) buffer SSBO_points {
	Vertex ssbo_points[];
};


void main(){

	uint index = gl_GlobalInvocationID.x;

	// if(index >= numPoints){
	// 	return;
	// }

	Vertex point = ssbo_points[index];

	vec4 pos_point = vec4(point.x, point.y, point.z, 1.0);

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

	uint depth = uint(-viewPos.z * ${depthPrecision});

	uint old_depth = atomicMin(ssbo_depth[pixelID], depth);

	if(depth < old_depth){

		//uint a = (point.color >> 24) & 0xFFu;
		uint a = 200;
		uint r = a * ((point.color >> 0) & 0xFFu);
		uint g = a * ((point.color >> 8) & 0xFFu);
		uint b = a * ((point.color >> 16) & 0xFFu);

		// atomicAdd(ssbo_color[4 * pixelID + 0], r);
		// atomicAdd(ssbo_color[4 * pixelID + 1], g);
		// atomicAdd(ssbo_color[4 * pixelID + 2], b);
		// atomicAdd(ssbo_color[4 * pixelID + 3], a);

		ssbo_color[4 * pixelID + 0] = r;
		ssbo_color[4 * pixelID + 1] = g;
		ssbo_color[4 * pixelID + 2] = b;
		ssbo_color[4 * pixelID + 3] = a;

		ssbo_point_id[pixelID] = point.pointID;
	}
}

`;