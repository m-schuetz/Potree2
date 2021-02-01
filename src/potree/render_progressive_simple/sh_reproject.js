
export let vs = `
[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
	[[offset(128)]] screen_width : f32;
	[[offset(132)]] screen_height : f32;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), group(0)]] var<uniform_constant> mySampler: sampler;
[[binding(2), group(0)]] var<uniform_constant> tex_color: texture_2d<f32>;
[[binding(3), group(0)]] var<uniform_constant> tex_pos: texture_2d<f32>;

[[builtin(position)]] var<out> out_pos : vec4<f32>;
[[builtin(vertex_index)]] var<in> vertexID : u32;

[[location(0)]] var<out> out_color : vec4<f32>;
[[location(1)]] var<out> out_vpos : vec4<f32>;



[[stage(vertex)]]
fn main() -> void {

	var x : u32 = vertexID % u32(uniforms.screen_width);
	var y : u32 = vertexID / u32(uniforms.screen_width);

	var u : f32 = f32(x) / uniforms.screen_width + 0.5 / uniforms.screen_width;
	var v : f32 = f32(y) / uniforms.screen_height + 0.5 / uniforms.screen_height;

	var uv : vec2<f32> = vec2<f32>(u, v);
	var color : vec4<f32> = textureSampleLevel(tex_color, mySampler, uv, 0);
	var pos : vec4<f32> = textureSampleLevel(tex_pos, mySampler, uv, 0);

	var viewPos : vec4<f32> = uniforms.worldView * pos;
	out_pos = uniforms.proj * viewPos;
	out_vpos = pos;

	if(pos.x == 0.0){
		out_pos = vec4<f32>(10.0, 10.0, 10.0, 1.0);
	}
	
	out_color = color;
	out_color.a = 1.0;
	//out_color = vec4<f32>(0.0, 1.0, 0.0, 1.0);

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