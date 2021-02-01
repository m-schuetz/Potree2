
let far = "20000.0";

export let vs = `
[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
	[[offset(128)]] screen_width : f32;
	[[offset(132)]] screen_height : f32;
};

[[block]] struct U32s {
	[[offset(0)]] values : [[stride(4)]] array<u32>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), set(0)]] var<storage_buffer> ssbo_attribute : [[access(read)]]U32s;

[[location(0)]] var<in> in_position : vec4<f32>;

[[builtin(instance_index)]] var<in> instanceIdx : i32;
[[builtin(vertex_index)]] var<in> vertexID : u32;
[[builtin(position)]] var<out> out_pos : vec4<f32>;

[[location(0)]] var<out> out_color : vec4<f32>;
[[location(1)]] var<out> out_vpos : vec4<f32>;

fn readU8(offset : u32) -> u32{
	var ipos : u32 = offset / 4u;
	var val_u32 : u32 = ssbo_attribute.values[ipos];

	var shift : u32 = 8u * (3u - (offset % 4u));
	var val_u8 : u32 = (val_u32 >> shift) & 0xFFu;

	return val_u8;
}

fn readU16(offset : u32) -> u32{
	var ipos : u32 = offset / 4u;
	var value : u32 = ssbo_attribute.values[ipos];

	if((offset & 2u) > 0u){
		value = (value >> 16) & 0xFFFFu;
	}else{
		value = value & 0xFFFFu;
	}

	return value;
}

fn getColor() -> vec4<f32>{

	// var R : u32 = readU8(4u * vertexID + 1u);
	// var G : u32 = readU8(4u * vertexID + 2u);
	// var B : u32 = readU8(4u * vertexID + 3u);

	var R : u32 = readU16(6u * vertexID + 0u);
	var G : u32 = readU16(6u * vertexID + 2u);
	var B : u32 = readU16(6u * vertexID + 4u);

	var r : f32 = f32(R) / (256.0 * 256.0);
	var g : f32 = f32(G) / (256.0 * 256.0);
	var b : f32 = f32(B) / (256.0 * 256.0);

	var color : vec4<f32> = vec4<f32>(r, g, b, 1.0);

	return color;
}

[[stage(vertex)]]
fn main() -> void {

	var viewPos : vec4<f32> = uniforms.worldView * in_position;
	out_pos = uniforms.proj * viewPos;
	out_vpos = in_position;

	// out_pos = out_pos / out_pos.w;
	
	out_pos.x = out_pos.x / out_pos.w;
	out_pos.y = out_pos.y / out_pos.w;
	out_pos.z = (-viewPos.z) / ${far};
	out_pos.w = 1.0;

	out_color = getColor();

	return;
}
`;

export let fs = `
[[location(0)]] var<in> in_color : vec4<f32>;
[[location(1)]] var<in> in_pos : vec4<f32>;

[[location(0)]] var<out> out_color : vec4<f32>;
[[location(1)]] var<out> out_pos : vec4<f32>;

[[stage(fragment)]]
fn main() -> void {
	out_color = in_color;
	out_pos = in_pos;

	return;
}
`;