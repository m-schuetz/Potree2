
let far = "20000.0";

export let vs = `
const pos : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
	vec2<f32>(-1.0, -1.0),
	vec2<f32>( 1.0, -1.0),
	vec2<f32>( 1.0,  1.0),
	vec2<f32>(-1.0, -1.0),
	vec2<f32>( 1.0,  1.0),
	vec2<f32>(-1.0,  1.0)
);

[[builtin(position)]] var<out> out_pos : vec4<f32>;
[[builtin(vertex_index)]] var<in> vertexID : i32;

[[stage(vertex)]]
fn main() -> void {
	out_pos = vec4<f32>(pos[vertexID], 0.0, 1.0);

	return;
}

`;

export let fs = `

[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
	[[offset(128)]] screen_width : f32;
	[[offset(132)]] screen_height : f32;
};

[[location(0)]] var<out> out_color : vec4<f32>;

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), group(0)]] var<uniform_constant> mySampler: sampler;
[[binding(2), group(0)]] var<uniform_constant> tex_color: texture_2d<f32>;
[[binding(3), set(0)]] var<uniform_constant> tex_depth: texture_sampled_2d<f32>;

[[builtin(frag_coord)]] var<in> fragCoord : vec4<f32>;

[[stage(fragment)]]
fn main() -> void {

	var uv : vec2<f32> = vec2<f32>(
		fragCoord.x / uniforms.screen_width,
		fragCoord.y / uniforms.screen_height
	);
	
	var color : vec4<f32> = textureSampleLevel(tex_color, mySampler, uv, 0);

	out_color = color;


	var width : f32 = uniforms.screen_width;
	var height : f32 = uniforms.screen_height;

	var reference_depth : f32 = 100.0;

	var sum : vec4<f32>;
	var window : f32 = 1.0;
	var window_radius : f32 = sqrt(2.0 * window * window);

	for(var i : f32 = -window; i <= window; i = i + 1.0){
		for(var j : f32 = -window; j <= window; j = j + 1.0){

			var x : f32 = clamp(fragCoord.x + i, 0.0, width - 1.0);
			var y : f32 = clamp(fragCoord.y + j, 0.0, height - 1.0);

			var uv : vec2<f32> = vec2<f32>(
				x / uniforms.screen_width,
				y / uniforms.screen_height
			);

			var depth : f32 = textureSampleLevel(tex_depth, mySampler, uv, 0).r;

			reference_depth = min(reference_depth, depth);
		}
	}

	reference_depth = reference_depth * 1.01;


	for(var i : f32 = -window; i <= window; i = i + 1.0){
		for(var j : f32 = -window; j <= window; j = j + 1.0){

			var x : f32 = clamp(fragCoord.x + i, 0.0, width - 1.0);
			var y : f32 = clamp(fragCoord.y + j, 0.0, height - 1.0);

			var uv : vec2<f32> = vec2<f32>(
				x / uniforms.screen_width,
				y / uniforms.screen_height
			);

			var depth : f32 = textureSampleLevel(tex_depth, mySampler, uv, 0).r;

			if(depth <= reference_depth){

				var dist : f32 = length(vec2<f32>(i, j)) / window_radius;
				var weight : f32 = max(0.0, 1.0 - dist);
				weight = pow(weight, 1.5);

				var color : vec4<f32> = textureSampleLevel(tex_color, mySampler, uv, 0);

				color.r = color.r * weight;
				color.g = color.g * weight;
				color.b = color.b * weight;
				color.a = color.a * weight;

				sum = sum + color;
			}

		}
	}

	var avg : vec4<f32> = vec4<f32>(
		sum.r / sum.a,
		sum.g / sum.a,
		sum.b / sum.a,
		1.0
	);

	out_color = avg;

	return;
}
`;