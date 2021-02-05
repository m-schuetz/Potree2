
let depthPrecision = "1000.0";

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
		[[offset(28)]] fillWindow : u32;
	};
	[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

	[[binding(1), set(0)]] var<storage_buffer> ssbo_colors : Colors;
	[[binding(2), set(0)]] var<storage_buffer> ssbo_depth : U32s;

	[[location(0)]] var<out> outColor : vec4<f32>;

	[[location(0)]] var<in> fragUV: vec2<f32>;

	[[builtin(frag_coord)]] var<in> fragCoord : vec4<f32>;

	[[stage(fragment)]]
	fn main() -> void {

		var avg : vec4<f32>;
		//var window : i32 = 3;
		var window : i32 = i32(uniforms.fillWindow);

		var width : i32 = i32(uniforms.screenWidth);
		var height : i32 = i32(uniforms.screenHeight);
		var frag_x : i32 = i32(fragCoord.x);
		var frag_y : i32 = i32(fragCoord.y);

		var cLeft : f32 = 0.0;
		var cRight : f32 = 0.0;
		var cTop : f32 = 0.0;
		var cBottom : f32 = 0.0;

		var referenceDepth : u32 = 0xffffffu;
		for(var i : i32 = -window; i <= window; i = i + 1){
			for(var j : i32 = -window; j <= window; j = j + 1){

				var x : i32 = clamp(frag_x + i, 0, width - 1);
				var y : i32 = clamp(height - frag_y - 1 + j, 0, height - 1);
				var index : u32 = u32(x + y * width);

				var depth : u32 = ssbo_depth.values[index];
				referenceDepth = min(referenceDepth, depth);
			}
		}
		referenceDepth = u32(f32(referenceDepth) * 1.01);

		for(var i : i32 = -window; i <= window; i = i + 1){
			for(var j : i32 = -window; j <= window; j = j + 1){

				var x : i32 = clamp(frag_x + i, 0, width - 1);
				var y : i32 = clamp(height - frag_y - 1 + j, 0, height - 1);
				var index : u32 = u32(x + y * width);

				var depth : u32 = ssbo_depth.values[index];

				if(depth > referenceDepth){
					continue;
				}

				// var r : u32 = ssbo_colors.values[4 * index + 0];
				// var g : u32 = ssbo_colors.values[4 * index + 1];
				// var b : u32 = ssbo_colors.values[4 * index + 2];
				// var c : u32 = ssbo_colors.values[4 * index + 3];

				var rg : u32 = ssbo_colors.values[2 * index + 0];
				var bc : u32 = ssbo_colors.values[2 * index + 1];
				var r : u32 = rg >> 16;
				var g : u32 = rg & 0xFFFF;
				var b : u32 = bc >> 16;
				var c : u32 = bc & 0xFFFF;

				var denom : f32 = f32(abs(i) + abs(j)) + 1.0;

				avg.x = avg.x + f32(r) / pow(denom, 4.0);
				avg.y = avg.y + f32(g) / pow(denom, 4.0);
				avg.z = avg.z + f32(b) / pow(denom, 4.0);
				avg.w = avg.w + f32(c) / pow(denom, 4.0);

			}
		}

		{
			var x : i32 = clamp(frag_x, 0, width - 1);
			var y : i32 = clamp(height - frag_y - 1, 0, height - 1);
			var index : u32 = u32(x + y * width);

			ssbo_depth.values[index] = 0xFFFFFFFFu;
			ssbo_colors.values[2 * index + 0] = 0u;
			ssbo_colors.values[2 * index + 1] = 0u;
		}

		avg.r = avg.r / avg.a;
		avg.g = avg.g / avg.a;
		avg.b = avg.b / avg.a;

		if(avg.a == 0.0){
			discard;
		}else{
			outColor.r = avg.r / 256.0;
			outColor.g = avg.g / 256.0;
			outColor.b = avg.b / 256.0;
			outColor.a = 1.0;
		}

		outColor.a = 1.0;


	}
`;