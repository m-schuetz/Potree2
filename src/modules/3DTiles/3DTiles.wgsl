
struct Uniforms {
	worldView        : mat4x4f,
	proj             : mat4x4f,
	screen_width     : f32,
	screen_height    : f32,
	size             : f32,
	elementCounter   : u32,
	hoveredIndex     : i32,
};

struct Node{
	worldView       : mat4x4f,
	ptr_indexBuffer : u32,
	ptr_posBuffer   : u32,
	ptr_uvBuffer    : u32,
	index           : u32,
	counter         : u32,
	pad_0           : u32,
	pad_1           : u32,
	pad_2           : u32,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage> nodes : array<Node>;

@group(1) @binding(0) var<storage> buffer : array<u32>;
@group(1) @binding(1) var _texture        : texture_2d<f32>;
@group(1) @binding(2) var _sampler        : sampler;





fn readU8(offset : u32) -> u32{
	var ipos    = offset / 4u;
	var val_u32 = buffer[ipos];
	var shift   = 8u * (offset % 4u);

	var val_u8  = (val_u32 >> shift) & 0xFFu;

	return val_u8;
}

fn readU16(offset : u32) -> u32{
	
	var first = readU8(offset + 0u);
	var second = readU8(offset + 1u);

	var value = first | (second << 8u);

	return value;
}

fn readI16(offset : u32) -> i32{
	
	var first = u32(readU8(offset + 0u));
	var second = u32(readU8(offset + 1u));

	var sign = second >> 7u;
	second = second & 127u;

	var value = -2;

	if(sign == 0u){
		value = 0;
	}else{
		value = -1;
	}

	var mask = 0xffff0000u;
	value = value | i32(first << 0u);
	value = value | i32(second << 8u);

	return value;
}

fn readU32(offset : u32) -> u32{
	
	var d0 = readU8(offset + 0u);
	var d1 = readU8(offset + 1u);
	var d2 = readU8(offset + 2u);
	var d3 = readU8(offset + 3u);

	var value = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	return value;
}

fn readI32(offset : u32) -> i32{
	
	var d0 = readU8(offset + 0u);
	var d1 = readU8(offset + 1u);
	var d2 = readU8(offset + 2u);
	var d3 = readU8(offset + 3u);

	var value = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	return i32(value);
}

fn readF32(offset : u32) -> f32{
	
	var d0 = readU8(offset + 0u);
	var d1 = readU8(offset + 1u);
	var d2 = readU8(offset + 2u);
	var d3 = readU8(offset + 3u);

	var value_u32 = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	var value_f32 = bitcast<f32>(value_u32);

	return value_f32;
}



struct VertexIn{
	@builtin(vertex_index) vertex_index : u32,
	@builtin(instance_index) instance_index : u32,
};

struct VertexOut{
	@builtin(position) position : vec4<f32>,
	@location(0) @interpolate(flat) pointID : u32,
	@location(1) @interpolate(linear) color : vec4<f32>,
	@location(2) @interpolate(perspective) uv : vec2f,
	@location(3) @interpolate(flat)  instanceID : u32,
};

struct FragmentIn{
	@location(0) @interpolate(flat) pointID : u32,
	@location(1) @interpolate(linear) color : vec4<f32>,
	@location(2) @interpolate(perspective) uv : vec2f,
	@location(3) @interpolate(flat)  instanceID : u32,
};

struct FragmentOut{
	@location(0) color : vec4<f32>,
	@location(1) point_id : u32,
};

@vertex
fn main_vertex(vertex : VertexIn) -> VertexOut {

	var node = nodes[vertex.instance_index];

	var vertexIndex = readU16(node.ptr_indexBuffer + 2u * vertex.vertex_index);
	var triangleIndex = vertex.vertex_index / 3u;

	var pos = vec4f(
		readF32(node.ptr_posBuffer + 12u * vertexIndex + 0u),
		-readF32(node.ptr_posBuffer + 12u * vertexIndex + 8u),
		readF32(node.ptr_posBuffer + 12u * vertexIndex + 4u),
		1.0,
	);

	var uv = vec2f(
		readF32(node.ptr_uvBuffer + 8u * vertexIndex + 0u),
		readF32(node.ptr_uvBuffer + 8u * vertexIndex + 4u),
	);

	var vout = VertexOut();
	vout.position = uniforms.proj * node.worldView * pos;
	vout.pointID = node.counter + triangleIndex;
	vout.uv = uv;
	vout.instanceID = vertex.instance_index;

	return vout;
}

@fragment
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout = FragmentOut();
	fout.point_id = uniforms.elementCounter + fragment.pointID;

	const SPECTRAL = array(
		vec3f(158.0,   1.0,  66.0),
		vec3f(213.0,  62.0,  79.0),
		vec3f(244.0, 109.0,  67.0),
		vec3f(253.0, 174.0,  97.0),
		vec3f(254.0, 224.0, 139.0),
		vec3f(255.0, 255.0, 191.0),
		vec3f(230.0, 245.0, 152.0),
		vec3f(171.0, 221.0, 164.0),
		vec3f(102.0, 194.0, 165.0),
		vec3f( 50.0, 136.0, 189.0),
		vec3f( 94.0,  79.0, 162.0),
	);

	var node = nodes[fragment.instanceID];

	// fout.color.r = f32(node.index % 10u) / 10.0;

	var color = SPECTRAL[node.index % 10u];
	fout.color.r = color.r / 256.0;
	fout.color.g = color.g / 256.0;
	fout.color.b = color.b / 256.0;

	fout.color = vec4f(1.0, 1.0, 1.0, 1.0);

	fout.color = vec4f(
		fragment.uv.x, 
		fragment.uv.y,
		0.0, 1.0
	);

	var c = textureSample(_texture, _sampler, fragment.uv);

	fout.color = c;

	return fout;
}
