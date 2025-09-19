
struct Uniforms {	
	worldView        : mat4x4f,
	world            : mat4x4f,
	view             : mat4x4f,
	proj             : mat4x4f,
	screen_width     : f32,
	screen_height    : f32,
	size             : f32,
	elementCounter   : u32,
	hoveredIndex     : i32,
	numSplats        : u32,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage> a_positions : array<f32>;
@group(0) @binding(2) var<storage, read_write> o_keys : array<u32>;
@group(0) @binding(3) var<storage, read_write> o_values : array<u32>;

@compute @workgroup_size(16,16)
fn main_vertex(
	@builtin(global_invocation_id) id: vec3<u32>,
	@builtin(num_workgroups) numWorkgroups: vec3<u32>,
	@builtin(workgroup_id) workgroupCoord: vec3<u32>,
	@builtin(local_invocation_index) localIndex: u32,
) {

	var workgroupIndex = workgroupCoord.x + workgroupCoord.y * numWorkgroups.x;
	var threadIndex = 16u * 16u * workgroupIndex + localIndex;
	var splatIndex = threadIndex;

	if(splatIndex >= uniforms.numSplats){ 
		return; 
	}

	var splatPos =  vec4f(
		a_positions[3u * splatIndex + 0], 
		a_positions[3u * splatIndex + 1], 
		a_positions[3u * splatIndex + 2], 
		1.0f);
	var worldPos = uniforms.world * splatPos;
	var viewPos = uniforms.view * worldPos;

	var depth = -viewPos.z;
	var value = splatIndex;

	// depth = 10000.0f - depth;
	// depth = f32(splatIndex);

	o_keys[splatIndex] = bitcast<u32>(depth);
	o_values[splatIndex] = splatIndex;
}
