
export let fsQuad = `

	[[block]] struct Colors {
		[[offset(0)]] values : [[stride(4)]] array<u32>;
	};

	[[block]] struct U32s {
		[[offset(0)]] values : [[stride(4)]] array<u32>;
	};

	[[block]] struct Uniforms {
		[[offset(0)]] uTest : u32;
		[[offset(4)]] x : f32;
		[[offset(8)]] y : f32;
		[[offset(12)]] width : f32;
		[[offset(16)]] height : f32;
		[[offset(20)]] screenWidth : f32;
		[[offset(24)]] screenHeight : f32;
	};
	[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

	[[binding(1), set(0)]] var<storage_buffer> ssbo_colors : [[access(read)]]Colors;

	[[location(0)]] var<out> outColor : vec4<f32>;

	[[location(0)]] var<in> fragUV: vec2<f32>;

	[[builtin(frag_coord)]] var<in> fragCoord : vec4<f32>;

	[[stage(fragment)]]
	fn main() -> void {

		var width : i32 = i32(uniforms.screenWidth);
		var height : i32 = i32(uniforms.screenHeight);
		var x : i32 = i32(fragCoord.x);
		var y : i32 = height - i32(fragCoord.y) - 1;

		var index : u32 = u32(x + y * width);

		var color : u32 = ssbo_colors.values[index];

		outColor.r = f32((color >> 0) & 0xFFu) / 255.0;
		outColor.g = f32((color >> 8) & 0xFFu) / 255.0;
		outColor.b = f32((color >> 16) & 0xFFu) / 255.0;
		outColor.a = 1.0;

	}
`;