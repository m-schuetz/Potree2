
import {Vector3, Matrix4} from "../math/math.js";
import {RenderTarget} from "../core/RenderTarget.js";

// import glslangModule from "../../libs/glslang/glslang.js";


// let glslang = null;
// glslangModule().then( result => {
// 	glslang = result;
// });

let csDepth = `

#version 450

layout(local_size_x = 128, local_size_y = 1) in;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldView;
	mat4 proj;
	uint width;
	uint height;
} uniforms;


layout(std430, set = 0, binding = 1) buffer SSBO {
	uint framebuffer[];
};

layout(std430, set = 0, binding = 2) buffer SSBO_position {
	float positions[];
};

layout(std430, set = 0, binding = 3) buffer SSBO_color {
	uint colors[];
};



void main(){

	uint index = gl_GlobalInvocationID.x;

	vec4 pos_point = vec4(
		positions[3 * index + 0],
		positions[3 * index + 1],
		positions[3 * index + 2],
		1.0);

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

	uint color = colors[index];

	//uint r = (color >> 0) & 0xFFu;
	//uint g = (color >> 8) & 0xFFu;
	//uint b = (color >> 16) & 0xFFu;
	//uint a = 255u;
	//uint c = (r << 24) | (g << 16) | (b << 8) | a;

	//framebuffer[pixelID] = c;

	uint depth = uint(-viewPos.z * 1000.0);

	atomicMin(framebuffer[pixelID], depth);
}

`;

let csColor = `

#version 450

layout(local_size_x = 128, local_size_y = 1) in;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldView;
	mat4 proj;
	uint width;
	uint height;
} uniforms;


layout(std430, set = 0, binding = 1) buffer SSBO_COLORS {
	uint ssbo_colors[];
};

layout(std430, set = 0, binding = 2) buffer SSBO_DEPTH {
	uint ssbo_depth[];
};

layout(std430, set = 0, binding = 3) buffer SSBO_position {
	float positions[];
};

layout(std430, set = 0, binding = 4) buffer SSBO_color {
	uint colors[];
};



void main(){

	uint index = gl_GlobalInvocationID.x;

	vec4 pos_point = vec4(
		positions[3 * index + 0],
		positions[3 * index + 1],
		positions[3 * index + 2],
		1.0);

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

	uint color = colors[index];

	uint r = (color >> 0) & 0xFFu;
	uint g = (color >> 8) & 0xFFu;
	uint b = (color >> 16) & 0xFFu;

	uint depth = uint(-viewPos.z * 1000.0);
	uint bufferedDepth = ssbo_depth[pixelID];

	if(depth < 1.01 * bufferedDepth){
		atomicAdd(ssbo_colors[4 * pixelID + 0], r);
		atomicAdd(ssbo_colors[4 * pixelID + 1], g);
		atomicAdd(ssbo_colors[4 * pixelID + 2], b);
		atomicAdd(ssbo_colors[4 * pixelID + 3], 1);
	}
}

`;

let csReset = `

#version 450

layout(local_size_x = 128, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer SSBO {
	uint framebuffer[];
};

layout(set = 0, binding = 1) uniform Uniforms {
	uint value;
} uniforms;

void main(){

	uint index = gl_GlobalInvocationID.x;

	framebuffer[index] = uniforms.value;
}
`;



let vs = `
	const pos : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(0.1, 0.0),
		vec2<f32>(0.1, 0.1),
		vec2<f32>(0.0, 0.0),
		vec2<f32>(0.1, 0.1),
		vec2<f32>(0.0, 0.1)
	);

	const uv : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
		vec2<f32>(0.0, 1.0),
		vec2<f32>(1.0, 1.0),
		vec2<f32>(1.0, 0.0),
		vec2<f32>(0.0, 1.0),
		vec2<f32>(1.0, 0.0),
		vec2<f32>(0.0, 0.0)
	);

	[[builtin(position)]] var<out> Position : vec4<f32>;
	[[builtin(vertex_idx)]] var<in> VertexIndex : i32;

	[[block]] struct Uniforms {
		[[offset(0)]] uTest : u32;
		[[offset(4)]] x : f32;
		[[offset(8)]] y : f32;
		[[offset(12)]] width : f32;
		[[offset(16)]] height : f32;
	};
	[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

	[[location(0)]] var<out> fragUV : vec2<f32>;

	[[stage(vertex)]]
	fn main() -> void {
		Position = vec4<f32>(pos[VertexIndex], 0.999, 1.0);
		fragUV = uv[VertexIndex];

		var x : f32 = uniforms.x * 2.0 - 1.0;
		var y : f32 = uniforms.y * 2.0 - 1.0;
		var width : f32 = uniforms.width * 2.0;
		var height : f32 = uniforms.height * 2.0;

		if(VertexIndex == 0){
			Position.x = x;
			Position.y = y;
		}elseif(VertexIndex == 1){
			Position.x = x + width;
			Position.y = y;
		}elseif(VertexIndex == 2){
			Position.x = x + width;
			Position.y = y + height;
		}elseif(VertexIndex == 3){
			Position.x = x;
			Position.y = y;
		}elseif(VertexIndex == 4){
			Position.x = x + width;
			Position.y = y + height;
		}elseif(VertexIndex == 5){
			Position.x = x;
			Position.y = y + height;
		}

		return;
	}
`;

let fs = `

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

	[[binding(1), set(0)]] var<storage_buffer> ssbo_colors : Colors;
	[[binding(2), set(0)]] var<storage_buffer> ssbo_depth : U32s;

	[[location(0)]] var<out> outColor : vec4<f32>;

	[[location(0)]] var<in> fragUV: vec2<f32>;

	[[builtin(frag_coord)]] var<in> fragCoord : vec4<f32>;

	[[stage(fragment)]]
	fn main() -> void {

		var avg : vec4<f32>;
		var window : i32 = 2;

		

		var referenceDepth : u32 = 0xffffffu;
		for(var i : i32 = -window; i <= window; i = i + 1){
			for(var j : i32 = -window; j <= window; j = j + 1){
				
				var index : u32 = 
					u32(fragCoord.x) +
					(u32(uniforms.screenHeight) - u32(fragCoord.y) - 1) * u32(uniforms.screenWidth) 
					+ i + j * i32(uniforms.screenWidth);

				var depth : u32 = ssbo_depth.values[index];
				referenceDepth = min(referenceDepth, depth);
			}
		}
		referenceDepth = u32(f32(referenceDepth) * 1.01);

		for(var i : i32 = -window; i <= window; i = i + 1){
			for(var j : i32 = -window; j <= window; j = j + 1){

				var index : u32 = 
					u32(fragCoord.x) +
					(u32(uniforms.screenHeight) - u32(fragCoord.y) - 1) * u32(uniforms.screenWidth) 
					+ i + j * i32(uniforms.screenWidth);

				var depth : u32 = ssbo_depth.values[index];

				if(depth > referenceDepth){
					continue;
				}

				var r : u32 = ssbo_colors.values[4 * index + 0];
				var g : u32 = ssbo_colors.values[4 * index + 1];
				var b : u32 = ssbo_colors.values[4 * index + 2];
				var c : u32 = ssbo_colors.values[4 * index + 3];

				var denom : f32 = f32(abs(i) + abs(j)) + 1.0;

				avg.x = avg.x + f32(r) / pow(denom, 3.0);
				avg.y = avg.y + f32(g) / pow(denom, 3.0);
				avg.z = avg.z + f32(b) / pow(denom, 3.0);
				avg.w = avg.w + f32(c) / pow(denom, 3.0);
			}
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

#		var index : u32 = 
#			u32(fragCoord.x) +
#			(u32(uniforms.screenHeight) - u32(fragCoord.y) - 1) * u32(uniforms.screenWidth);
#		
#		var c : u32 = ssbo_colors.values[4 * index + 3];
#
#		if(c == 0){
#			discard;
#		}else{
#			var r : u32 = ssbo_colors.values[4 * index + 0] / c;
#			var g : u32 = ssbo_colors.values[4 * index + 1] / c;
#			var b : u32 = ssbo_colors.values[4 * index + 2] / c;
#
#			outColor.r = f32(r) / 256.0;
#			outColor.g = f32(g) / 256.0;
#			outColor.b = f32(b) / 256.0;
#			outColor.a = 1.0;
#		}

	}
`;




let _target_1 = null;

function getTarget1(renderer){
	if(_target_1 === null){

		let size = [128, 128, 1];
		_target_1 = new RenderTarget(renderer, {
			size: size,
			colorDescriptors: [{
				size: size,
				format: renderer.swapChainFormat,
				usage: GPUTextureUsage.SAMPLED 
					// | GPUTextureUsage.COPY_SRC 
					// | GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.OUTPUT_ATTACHMENT,
			}],
			depthDescriptor: {
				size: size,
				format: "depth24plus-stencil8",
				usage: GPUTextureUsage.SAMPLED 
					// | GPUTextureUsage.COPY_SRC 
					// | GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.OUTPUT_ATTACHMENT,
			}
		});
	}

	return _target_1;
}


let depthState = null;
let colorState = null;
let resetState = null;
let screenPassState = null;

function getDepthState(renderer){

	if(!depthState){

		let {device} = renderer;

		// let target = getTarget1(renderer);
		let ssboSize = 2560 * 1440 * 4 * 4;
		let ssbo = renderer.createBuffer(ssboSize);

		//let compiled = glslang.compileGLSL(csDepth, "compute");
		//console.log("compiled: ", compiled.join(", "));

		let compiled = new Uint32Array([119734787, 65536, 524296, 173, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0, 196622, 0, 1, 393231, 5, 4, 1852399981, 0, 11, 393232, 4, 17, 128, 1,
			1, 196611, 2, 450, 262149, 4, 1852399981, 0, 262149, 8, 1701080681, 120, 524293, 11, 1197436007, 1633841004, 1986939244, 1952539503, 1231974249, 68, 327685, 19,
			1601400688, 1852403568, 116, 393221, 21, 1329746771, 1936683103, 1869182057, 110, 393222, 21, 0, 1769172848, 1852795252, 115, 196613, 23, 0, 262149, 47,
			2003134838, 7565136, 327685, 49, 1718185557, 1936552559, 0, 393222, 49, 0, 1819438967, 1701402212, 119, 327686, 49, 1, 1785688688, 0, 327686, 49, 2, 1952737655,
			104, 327686, 49, 3, 1734960488, 29800, 327685, 51, 1718185589, 1936552559, 0, 196613, 57, 7565168, 327685, 112, 1734438249, 2053722981, 101, 262149, 125,
			1348955497, 29551, 327685, 135, 1702390128, 1869562732, 7562354, 262149, 139, 1702390128, 4475244, 262149, 148, 1869377379, 114, 327685, 150, 1329746771,
			1819239263, 29295, 327686, 150, 0, 1869377379, 29554, 196613, 152, 0, 262149, 156, 1953523044, 104, 262149, 164, 1329746771, 0, 393222, 164, 0, 1835102822,
			1718968933, 7497062, 196613, 166, 0, 262215, 11, 11, 28, 262215, 20, 6, 4, 327752, 21, 0, 35, 0, 196679, 21, 3, 262215, 23, 34, 0, 262215, 23, 33, 2, 262216, 49,
			0, 5, 327752, 49, 0, 35, 0, 327752, 49, 0, 7, 16, 262216, 49, 1, 5, 327752, 49, 1, 35, 64, 327752, 49, 1, 7, 16, 327752, 49, 2, 35, 128, 327752, 49, 3, 35, 132,
			196679, 49, 2, 262215, 51, 34, 0, 262215, 51, 33, 0, 262215, 149, 6, 4, 327752, 150, 0, 35, 0, 196679, 150, 3, 262215, 152, 34, 0, 262215, 152, 33, 3, 262215,
			163, 6, 4, 327752, 164, 0, 35, 0, 196679, 164, 3, 262215, 166, 34, 0, 262215, 166, 33, 1, 262215, 172, 11, 25, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0, 262176,
			7, 7, 6, 262167, 9, 6, 3, 262176, 10, 1, 9, 262203, 10, 11, 1, 262187, 6, 12, 0, 262176, 13, 1, 6, 196630, 16, 32, 262167, 17, 16, 4, 262176, 18, 7, 17, 196637,
			20, 16, 196638, 21, 20, 262176, 22, 2, 21, 262203, 22, 23, 2, 262165, 24, 32, 1, 262187, 24, 25, 0, 262187, 6, 26, 3, 262176, 30, 2, 16, 262187, 6, 35, 1, 262187,
			6, 41, 2, 262187, 16, 45, 1065353216, 262168, 48, 17, 4, 393246, 49, 48, 48, 6, 6, 262176, 50, 2, 49, 262203, 50, 51, 2, 262176, 52, 2, 48, 262187, 24, 58, 1,
			262167, 63, 16, 3, 262176, 66, 7, 16, 131092, 73, 262187, 16, 76, 0, 262187, 16, 83, 3212836864, 262167, 110, 24, 2, 262176, 111, 7, 110, 262187, 24, 113, 2,
			262176, 114, 2, 6, 262187, 24, 118, 3, 262167, 123, 16, 2, 262176, 124, 7, 123, 262187, 16, 128, 1056964608, 262176, 138, 7, 24, 196637, 149, 6, 196638, 150, 149,
			262176, 151, 2, 150, 262203, 151, 152, 2, 262187, 16, 160, 1148846080, 196637, 163, 6, 196638, 164, 163, 262176, 165, 2, 164, 262203, 165, 166, 2, 262187, 6, 171,
			128, 393260, 9, 172, 171, 35, 35, 327734, 2, 4, 0, 3, 131320, 5, 262203, 7, 8, 7, 262203, 18, 19, 7, 262203, 18, 47, 7, 262203, 18, 57, 7, 262203, 111, 112, 7,
			262203, 124, 125, 7, 262203, 111, 135, 7, 262203, 138, 139, 7, 262203, 7, 148, 7, 262203, 7, 156, 7, 327745, 13, 14, 11, 12, 262205, 6, 15, 14, 196670, 8, 15,
			262205, 6, 27, 8, 327812, 6, 28, 26, 27, 327808, 6, 29, 28, 12, 393281, 30, 31, 23, 25, 29, 262205, 16, 32, 31, 262205, 6, 33, 8, 327812, 6, 34, 26, 33, 327808,
			6, 36, 34, 35, 393281, 30, 37, 23, 25, 36, 262205, 16, 38, 37, 262205, 6, 39, 8, 327812, 6, 40, 26, 39, 327808, 6, 42, 40, 41, 393281, 30, 43, 23, 25, 42, 262205,
			16, 44, 43, 458832, 17, 46, 32, 38, 44, 45, 196670, 19, 46, 327745, 52, 53, 51, 25, 262205, 48, 54, 53, 262205, 17, 55, 19, 327825, 17, 56, 54, 55, 196670, 47,
			56, 327745, 52, 59, 51, 58, 262205, 48, 60, 59, 262205, 17, 61, 47, 327825, 17, 62, 60, 61, 196670, 57, 62, 262205, 17, 64, 57, 524367, 63, 65, 64, 64, 0, 1, 2,
			327745, 66, 67, 57, 26, 262205, 16, 68, 67, 393296, 63, 69, 68, 68, 68, 327816, 63, 70, 65, 69, 262205, 17, 71, 57, 589903, 17, 72, 71, 70, 4, 5, 6, 3, 196670,
			57, 72, 327745, 66, 74, 57, 26, 262205, 16, 75, 74, 327868, 73, 77, 75, 76, 262312, 73, 78, 77, 196855, 80, 0, 262394, 78, 79, 80, 131320, 79, 327745, 66, 81, 57,
			12, 262205, 16, 82, 81, 327864, 73, 84, 82, 83, 131321, 80, 131320, 80, 458997, 73, 85, 77, 5, 84, 79, 262312, 73, 86, 85, 196855, 88, 0, 262394, 86, 87, 88,
			131320, 87, 327745, 66, 89, 57, 12, 262205, 16, 90, 89, 327866, 73, 91, 90, 45, 131321, 88, 131320, 88, 458997, 73, 92, 85, 80, 91, 87, 262312, 73, 93, 92,
			196855, 95, 0, 262394, 93, 94, 95, 131320, 94, 327745, 66, 96, 57, 35, 262205, 16, 97, 96, 327864, 73, 98, 97, 83, 131321, 95, 131320, 95, 458997, 73, 99, 92, 88,
			98, 94, 262312, 73, 100, 99, 196855, 102, 0, 262394, 100, 101, 102, 131320, 101, 327745, 66, 103, 57, 35, 262205, 16, 104, 103, 327866, 73, 105, 104, 45, 131321,
			102, 131320, 102, 458997, 73, 106, 99, 95, 105, 101, 196855, 108, 0, 262394, 106, 107, 108, 131320, 107, 65789, 131320, 108, 327745, 114, 115, 51, 113, 262205, 6,
			116, 115, 262268, 24, 117, 116, 327745, 114, 119, 51, 118, 262205, 6, 120, 119, 262268, 24, 121, 120, 327760, 110, 122, 117, 121, 196670, 112, 122, 262205, 17,
			126, 57, 458831, 123, 127, 126, 126, 0, 1, 327822, 123, 129, 127, 128, 327760, 123, 130, 128, 128, 327809, 123, 131, 129, 130, 262205, 110, 132, 112, 262255, 123,
			133, 132, 327813, 123, 134, 131, 133, 196670, 125, 134, 262205, 123, 136, 125, 262254, 110, 137, 136, 196670, 135, 137, 327745, 138, 140, 135, 12, 262205, 24,
			141, 140, 327745, 138, 142, 135, 35, 262205, 24, 143, 142, 327745, 138, 144, 112, 12, 262205, 24, 145, 144, 327812, 24, 146, 143, 145, 327808, 24, 147, 141, 146,
			196670, 139, 147, 262205, 6, 153, 8, 393281, 114, 154, 152, 25, 153, 262205, 6, 155, 154, 196670, 148, 155, 327745, 66, 157, 47, 41, 262205, 16, 158, 157, 262271,
			16, 159, 158, 327813, 16, 161, 159, 160, 262253, 6, 162, 161, 196670, 156, 162, 262205, 24, 167, 139, 393281, 114, 168, 166, 25, 167, 262205, 6, 169, 156, 458989,
			6, 170, 168, 35, 12, 169, 65789, 65592]);

		let csDescriptor = {
			code: compiled,
			source: csDepth,
		};
		let csModule = device.createShaderModule(csDescriptor);

		let uniformBufferSize = 2 * 64 + 8;
		let uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let pipeline = device.createComputePipeline({
			computeStage: {
				module: csModule,
				entryPoint: "main",
			}
		});

		depthState = {pipeline, ssbo, ssboSize, uniformBuffer};
	}

	return depthState;
}

function getColorState(renderer){

	if(!colorState){

		let {device} = renderer;

		let ssboSize = 4 * 2560 * 1440 * 4 * 4;
		let ssbo_colors = renderer.createBuffer(ssboSize);

		//let compiled = glslang.compileGLSL(csColor, "compute");
		//console.log("compiled: ", compiled.join(", "));

		let compiled = new Uint32Array([119734787, 65536, 524296, 225, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0, 196622, 0, 1, 393231, 5, 4, 1852399981, 0, 11, 393232, 4, 17, 128,
			1, 1, 196611, 2, 450, 262149, 4, 1852399981, 0, 262149, 8, 1701080681, 120, 524293, 11, 1197436007, 1633841004, 1986939244, 1952539503, 1231974249, 68, 327685,
			19, 1601400688, 1852403568, 116, 393221, 21, 1329746771, 1936683103, 1869182057, 110, 393222, 21, 0, 1769172848, 1852795252, 115, 196613, 23, 0, 262149, 47,
			2003134838, 7565136, 327685, 49, 1718185557, 1936552559, 0, 393222, 49, 0, 1819438967, 1701402212, 119, 327686, 49, 1, 1785688688, 0, 327686, 49, 2, 1952737655,
			104, 327686, 49, 3, 1734960488, 29800, 327685, 51, 1718185589, 1936552559, 0, 196613, 57, 7565168, 327685, 112, 1734438249, 2053722981, 101, 262149, 125,
			1348955497, 29551, 327685, 135, 1702390128, 1869562732, 7562354, 262149, 139, 1702390128, 4475244, 262149, 148, 1869377379, 114, 327685, 150, 1329746771,
			1819239263, 29295, 327686, 150, 0, 1869377379, 29554, 196613, 152, 0, 196613, 156, 114, 196613, 161, 103, 196613, 166, 98, 262149, 171, 1953523044, 104, 393221,
			178, 1717990754, 1684370021, 1953523012, 104, 327685, 180, 1329746771, 1346716767, 18516, 393222, 180, 0, 1868723059, 1885693023, 26740, 196613, 182, 0, 327685,
			196, 1329746771, 1280263007, 5460559, 393222, 196, 0, 1868723059, 1819239263, 7565935, 196613, 198, 0, 262215, 11, 11, 28, 262215, 20, 6, 4, 327752, 21, 0, 35,
			0, 196679, 21, 3, 262215, 23, 34, 0, 262215, 23, 33, 3, 262216, 49, 0, 5, 327752, 49, 0, 35, 0, 327752, 49, 0, 7, 16, 262216, 49, 1, 5, 327752, 49, 1, 35, 64,
			327752, 49, 1, 7, 16, 327752, 49, 2, 35, 128, 327752, 49, 3, 35, 132, 196679, 49, 2, 262215, 51, 34, 0, 262215, 51, 33, 0, 262215, 149, 6, 4, 327752, 150, 0, 35,
			0, 196679, 150, 3, 262215, 152, 34, 0, 262215, 152, 33, 4, 262215, 179, 6, 4, 327752, 180, 0, 35, 0, 196679, 180, 3, 262215, 182, 34, 0, 262215, 182, 33, 2,
			262215, 195, 6, 4, 327752, 196, 0, 35, 0, 196679, 196, 3, 262215, 198, 34, 0, 262215, 198, 33, 1, 262215, 224, 11, 25, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0,
			262176, 7, 7, 6, 262167, 9, 6, 3, 262176, 10, 1, 9, 262203, 10, 11, 1, 262187, 6, 12, 0, 262176, 13, 1, 6, 196630, 16, 32, 262167, 17, 16, 4, 262176, 18, 7, 17, 
			196637, 20, 16, 196638, 21, 20, 262176, 22, 2, 21, 262203, 22, 23, 2, 262165, 24, 32, 1, 262187, 24, 25, 0, 262187, 6, 26, 3, 262176, 30, 2, 16, 262187, 6, 35,
			1, 262187, 6, 41, 2, 262187, 16, 45, 1065353216, 262168, 48, 17, 4, 393246, 49, 48, 48, 6, 6, 262176, 50, 2, 49, 262203, 50, 51, 2, 262176, 52, 2, 48, 262187, 
			24, 58, 1, 262167, 63, 16, 3, 262176, 66, 7, 16, 131092, 73, 262187, 16, 76, 0, 262187, 16, 83, 3212836864, 262167, 110, 24, 2, 262176, 111, 7, 110, 262187, 24,
			113, 2, 262176, 114, 2, 6, 262187, 24, 118, 3, 262167, 123, 16, 2, 262176, 124, 7, 123, 262187, 16, 128, 1056964608, 262176, 138, 7, 24, 196637, 149, 6, 196638, 
			150, 149, 262176, 151, 2, 150, 262203, 151, 152, 2, 262187, 6, 159, 255, 262187, 24, 163, 8, 262187, 24, 168, 16, 262187, 16, 175, 1148846080, 196637, 179, 6,
			196638, 180, 179, 262176, 181, 2, 180, 262203, 181, 182, 2, 262187, 16, 188, 1065437102, 196637, 195, 6, 196638, 196, 195, 262176, 197, 2, 196, 262203, 197, 198, 
			2, 262187, 24, 199, 4, 262187, 6, 223, 128, 393260, 9, 224, 223, 35, 35, 327734, 2, 4, 0, 3, 131320, 5, 262203, 7, 8, 7, 262203, 18, 19, 7, 262203, 18, 47, 7,
			262203, 18, 57, 7, 262203, 111, 112, 7, 262203, 124, 125, 7, 262203, 111, 135, 7, 262203, 138, 139, 7, 262203, 7, 148, 7, 262203, 7, 156, 7, 262203, 7, 161, 7, 
			262203, 7, 166, 7, 262203, 7, 171, 7, 262203, 7, 178, 7, 327745, 13, 14, 11, 12, 262205, 6, 15, 14, 196670, 8, 15, 262205, 6, 27, 8, 327812, 6, 28, 26, 27,
			327808, 6, 29, 28, 12, 393281, 30, 31, 23, 25, 29, 262205, 16, 32, 31, 262205, 6, 33, 8, 327812, 6, 34, 26, 33, 327808, 6, 36, 34, 35, 393281, 30, 37, 23, 25, 
			36, 262205, 16, 38, 37, 262205, 6, 39, 8, 327812, 6, 40, 26, 39, 327808, 6, 42, 40, 41, 393281, 30, 43, 23, 25, 42, 262205, 16, 44, 43, 458832, 17, 46, 32, 38,
			44, 45, 196670, 19, 46, 327745, 52, 53, 51, 25, 262205, 48, 54, 53, 262205, 17, 55, 19, 327825, 17, 56, 54, 55, 196670, 47, 56, 327745, 52, 59, 51, 58, 262205, 
			48, 60, 59, 262205, 17, 61, 47, 327825, 17, 62, 60, 61, 196670, 57, 62, 262205, 17, 64, 57, 524367, 63, 65, 64, 64, 0, 1, 2, 327745, 66, 67, 57, 26, 262205, 16,
			68, 67, 393296, 63, 69, 68, 68, 68, 327816, 63, 70, 65, 69, 262205, 17, 71, 57, 589903, 17, 72, 71, 70, 4, 5, 6, 3, 196670, 57, 72, 327745, 66, 74, 57, 26, 262205, 
			16, 75, 74, 327868, 73, 77, 75, 76, 262312, 73, 78, 77, 196855, 80, 0, 262394, 78, 79, 80, 131320, 79, 327745, 66, 81, 57, 12, 262205, 16, 82, 81,
			327864, 73, 84, 82, 83, 131321, 80, 131320, 80, 458997, 73, 85, 77, 5, 84, 79, 262312, 73, 86, 85, 196855, 88, 0, 262394, 86, 87, 88, 131320, 87, 327745, 66, 89, 
			57, 12, 262205, 16, 90, 89, 327866, 73, 91, 90, 45, 131321, 88, 131320, 88, 458997, 73, 92, 85, 80, 91, 87, 262312, 73, 93, 92, 196855, 95, 0, 262394, 93, 94,
			95, 131320, 94, 327745, 66, 96, 57, 35, 262205, 16, 97, 96, 327864, 73, 98, 97, 83, 131321, 95, 131320, 95, 458997, 73, 99, 92, 88, 98, 94, 262312, 73, 100, 99, 
			196855, 102, 0, 262394, 100, 101, 102, 131320, 101, 327745, 66, 103, 57, 35, 262205, 16, 104, 103, 327866, 73, 105, 104, 45, 131321, 102, 131320, 102, 458997,
			73, 106, 99, 95, 105, 101, 196855, 108, 0, 262394, 106, 107, 108, 131320, 107, 65789, 131320, 108, 327745, 114, 115, 51, 113, 262205, 6, 116, 115, 262268, 24,
			117, 116, 327745, 114, 119, 51, 118, 262205, 6, 120, 119, 262268, 24, 121, 120, 327760, 110, 122, 117, 121, 196670, 112, 122, 262205, 17, 126, 57, 458831, 123,
			127, 126, 126, 0, 1, 327822, 123, 129, 127, 128, 327760, 123, 130, 128, 128, 327809, 123, 131, 129, 130, 262205, 110, 132, 112, 262255, 123, 133, 132, 327813,
			123, 134, 131, 133, 196670, 125, 134, 262205, 123, 136, 125, 262254, 110, 137, 136, 196670, 135, 137, 327745, 138, 140, 135, 12, 262205, 24, 141, 140, 327745,
			138, 142, 135, 35, 262205, 24, 143, 142, 327745, 138, 144, 112, 12, 262205, 24, 145, 144, 327812, 24, 146, 143, 145, 327808, 24, 147, 141, 146, 196670, 139, 147,
			262205, 6, 153, 8, 393281, 114, 154, 152, 25, 153, 262205, 6, 155, 154, 196670, 148, 155, 262205, 6, 157, 148, 327874, 6, 158, 157, 25, 327879, 6, 160, 158, 159,
			196670, 156, 160, 262205, 6, 162, 148, 327874, 6, 164, 162, 163, 327879, 6, 165, 164, 159, 196670, 161, 165, 262205, 6, 167, 148, 327874, 6, 169, 167, 168,
			327879, 6, 170, 169, 159, 196670, 166, 170, 327745, 66, 172, 47, 41, 262205, 16, 173, 172, 262271, 16, 174, 173, 327813, 16, 176, 174, 175, 262253, 6, 177, 176,
			196670, 171, 177, 262205, 24, 183, 139, 393281, 114, 184, 182, 25, 183, 262205, 6, 185, 184, 196670, 178, 185, 262205, 6, 186, 171, 262256, 16, 187, 186, 262205,
			6, 189, 178, 262256, 16, 190, 189, 327813, 16, 191, 188, 190, 327864, 73, 192, 187, 191, 196855, 194, 0, 262394, 192, 193, 194, 131320, 193, 262205, 24, 200,
			139, 327812, 24, 201, 199, 200, 327808, 24, 202, 201, 25, 393281, 114, 203, 198, 25, 202, 262205, 6, 204, 156, 458986, 6, 205, 203, 35, 12, 204, 262205, 24, 206,
			139, 327812, 24, 207, 199, 206, 327808, 24, 208, 207, 58, 393281, 114, 209, 198, 25, 208, 262205, 6, 210, 161, 458986, 6, 211, 209, 35, 12, 210, 262205, 24, 212,
			139, 327812, 24, 213, 199, 212, 327808, 24, 214, 213, 113, 393281, 114, 215, 198, 25, 214, 262205, 6, 216, 166, 458986, 6, 217, 215, 35, 12, 216, 262205, 24,
			218, 139, 327812, 24, 219, 199, 218, 327808, 24, 220, 219, 118, 393281, 114, 221, 198, 25, 220, 458986, 6, 222, 221, 35, 12, 35, 131321, 194, 131320, 194, 65789,
 			65592]);

		let csDescriptor = {
			code: compiled,
			source: csColor,
		};
		let csModule = device.createShaderModule(csDescriptor);

		let uniformBufferSize = 2 * 64 + 8;
		let uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let pipeline = device.createComputePipeline({
			computeStage: {
				module: csModule,
				entryPoint: "main",
			}
		});

		colorState = {pipeline, ssbo_colors, ssboSize, uniformBuffer};
	}

	return colorState;
}

function getResetState(renderer){

	if(!resetState){

		let {device} = renderer;

		let uniformBufferSize = 4;
		let uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		//let compiled = glslang.compileGLSL(csReset, "compute");
		//console.log("compiled: ", compiled.join(", "));

		let compiled = new Uint32Array([
			 119734787, 65536, 524296, 33, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0, 196622, 0, 1, 393231, 5, 4, 1852399981, 0, 11, 393232, 4, 17, 128, 1,
			1, 196611, 2, 450, 262149, 4, 1852399981, 0, 262149, 8, 1701080681, 120, 524293, 11, 1197436007, 1633841004, 1986939244, 1952539503, 1231974249, 68, 262149, 17,
			1329746771, 0, 393222, 17, 0, 1835102822, 1718968933, 7497062, 196613, 19, 0, 327685, 23, 1718185557, 1936552559, 0, 327686, 23, 0, 1970037110, 101, 327685, 25,
			1718185589, 1936552559, 0, 262215, 11, 11, 28, 262215, 16, 6, 4, 327752, 17, 0, 35, 0, 196679, 17, 3, 262215, 19, 34, 0, 262215, 19, 33, 0, 327752, 23, 0, 35, 0,
			196679, 23, 2, 262215, 25, 34, 0, 262215, 25, 33, 1, 262215, 32, 11, 25, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0, 262176, 7, 7, 6, 262167, 9, 6, 3, 262176, 10,
			1, 9, 262203, 10, 11, 1, 262187, 6, 12, 0, 262176, 13, 1, 6, 196637, 16, 6, 196638, 17, 16, 262176, 18, 2, 17, 262203, 18, 19, 2, 262165, 20, 32, 1, 262187, 20,
			21, 0, 196638, 23, 6, 262176, 24, 2, 23, 262203, 24, 25, 2, 262176, 26, 2, 6, 262187, 6, 30, 128, 262187, 6, 31, 1, 393260, 9, 32, 30, 31, 31, 327734, 2, 4, 0, 3,
			131320, 5, 262203, 7, 8, 7, 327745, 13, 14, 11, 12, 262205, 6, 15, 14, 196670, 8, 15, 262205, 6, 22, 8, 327745, 26, 27, 25, 21, 262205, 6, 28, 27, 393281, 26, 29,
			19, 21, 22, 196670, 29, 28, 65789, 65592
		]);

		let csDescriptor = {
			code: compiled,
			source: csReset,
		};
		let csModule = device.createShaderModule(csDescriptor);

		let pipeline = device.createComputePipeline({
			computeStage: {
				module: csModule,
				entryPoint: "main",
			}
		});

		resetState = {pipeline, uniformBuffer};
	}

	return resetState;
}

function getScreenPassState(renderer){

	if(!screenPassState){
		let {device, swapChainFormat} = renderer;

		let bindGroupLayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				type: "uniform-buffer"
			},{
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT,
				type: "storage-buffer"
			},{
				binding: 2,
				visibility: GPUShaderStage.FRAGMENT,
				type: "storage-buffer"
			}]
		});

		let layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
		
		let pipeline = device.createRenderPipeline({
			layout: layout, 
			vertexStage: {
				module: device.createShaderModule({code: vs}),
				entryPoint: "main",
			},
			fragmentStage: {
				module: device.createShaderModule({code: fs}),
				entryPoint: "main",
			},
			primitiveTopology: "triangle-list",
			depthStencilState: {
					depthWriteEnabled: true,
					depthCompare: "less",
					format: "depth24plus-stencil8",
			},
			colorStates: [{
				format: swapChainFormat,
			}],
		});

		let uniformBufferSize = 32;
		let uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		screenPassState = {bindGroupLayout, pipeline, uniformBuffer}
	}

	return screenPassState;
}

let frame = 0;

function reset(renderer, ssbo, numUints, value){

	let {device} = renderer;
	let {pipeline, uniformBuffer} = getResetState(renderer);

	let bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: {
					buffer: ssbo,
					offset: 0,
					size: numUints * 4,
				}
			},{
				binding: 1,
				resource: {
					buffer: uniformBuffer,
				}
			}
		]
	});


	{ // uniform buffer
		let data = new Uint32Array([value]);
		device.defaultQueue.writeBuffer(
			uniformBuffer,
			0,
			data.buffer,
			data.byteOffset,
			data.byteLength
		);
	}


	const commandEncoder = device.createCommandEncoder();

	let passEncoder = commandEncoder.beginComputePass();

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	let groups = numUints / 128;
	passEncoder.dispatch(groups, 1, 1);
	passEncoder.endPass();

	device.defaultQueue.submit([commandEncoder.finish()]);
}


export function renderAtomicDilate(renderer, octree, camera){

	// if(!glslang){
	// 	console.log("glslang not yet initialized");

	// 	return;
	// }

	let {device} = renderer;
	let nodes = octree.visibleNodes;

	if(nodes.length === 0){
		return getTarget1(renderer).colorAttachments[0].texture;
	}

	{ // RESIZE RENDER TARGET
		let size = renderer.getSize();
		let target = getTarget1(renderer);
		target.setSize(size.width, size.height);
	}

	{ // UPDATE UNIFORMS

		let data = new ArrayBuffer(136);
		let f32 = new Float32Array(data);
		let view = new DataView(data);

		{ // transform
			let world = octree.world;
			let view = camera.view;
			let worldView = new Matrix4().multiplyMatrices(view, world);

			f32.set(worldView.elements, 0);
			f32.set(camera.proj.elements, 16);
		}

		{ // screen size
			let size = renderer.getSize();

			view.setUint32(128, size.width, true);
			view.setUint32(132, size.height, true);
		}

		{ // set depth pass uniforms
			let {uniformBuffer} = getDepthState(renderer);
			device.defaultQueue.writeBuffer(
				uniformBuffer, 0,
				data, data.byteOffset, data.byteLength
			);
		}

		{ // set color pass uniforms
			let {uniformBuffer} = getColorState(renderer);
			device.defaultQueue.writeBuffer(
				uniformBuffer, 0,
				data, data.byteOffset, data.byteLength
			);
		}

	}

	{ // RESET BUFFERS
		let size = renderer.getSize();
		let numUints = size.width * size.height;

		let {ssbo} = getDepthState(renderer);
		let {ssbo_colors} = getColorState(renderer);

		reset(renderer, ssbo, numUints, 0xffffff);
		reset(renderer, ssbo_colors, 4 * numUints, 0);
	}


	{ // DEPTH PASS
		let {pipeline, uniformBuffer} = getDepthState(renderer);
		let ssbo_depth = getDepthState(renderer).ssbo;

		const commandEncoder = device.createCommandEncoder();
		let passEncoder = commandEncoder.beginComputePass();

		passEncoder.setPipeline(pipeline);

		for(let node of nodes){
			let gpuBuffers = renderer.getGpuBuffers(node.geometry);

			let bindGroup = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: [
					{
						binding: 0, resource: {buffer: uniformBuffer}
					},{
						binding: 1, resource: {buffer: ssbo_depth}
					},{
						binding: 2, resource: {buffer: gpuBuffers[0].vbo}
					},{
						binding: 3, resource: {buffer: gpuBuffers[1].vbo}
					}
				]
			});
			
			passEncoder.setBindGroup(0, bindGroup);

			let groups = Math.floor(node.geometry.numElements / 128);
			passEncoder.dispatch(groups);
			
		}

		passEncoder.endPass();
		device.defaultQueue.submit([commandEncoder.finish()]);

	}

	{ // COLOR PASS
		let {pipeline, uniformBuffer, ssboSize} = getColorState(renderer);
		let {ssbo_colors} = getColorState(renderer);
		let ssbo_depth = getDepthState(renderer).ssbo;

		const commandEncoder = device.createCommandEncoder();
		let passEncoder = commandEncoder.beginComputePass();

		passEncoder.setPipeline(pipeline);

		for(let node of nodes){
			let gpuBuffers = renderer.getGpuBuffers(node.geometry);

			let bindGroup = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: [
					{
						binding: 0, resource: {buffer: uniformBuffer}
					},{
						binding: 1, resource: {buffer: ssbo_colors}
					},{
						binding: 2, resource: {buffer: ssbo_depth}
					},{
						binding: 3, resource: {buffer: gpuBuffers[0].vbo}
					},{
						binding: 4, resource: {buffer: gpuBuffers[1].vbo}
					}
				]
			});

			passEncoder.setBindGroup(0, bindGroup);

			let groups = Math.floor(node.geometry.numElements / 128);
			passEncoder.dispatch(groups, 1, 1);
		}

		passEncoder.endPass();
		device.defaultQueue.submit([commandEncoder.finish()]);

	}

	{ // resolve
		let ssbo_depth = getDepthState(renderer).ssbo;
		let {ssbo_colors} = getColorState(renderer);
		let {pipeline, uniformBuffer} = getScreenPassState(renderer);
		let target = getTarget1(renderer);
		let size = renderer.getSize();

		let uniformBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [{
					binding: 0,
					resource: {buffer: uniformBuffer}
				},{
					binding: 1,
					resource: {buffer: ssbo_colors},
				},{
					binding: 2,
					resource: {buffer: ssbo_depth},
				}],
		});

		let renderPassDescriptor = {
			colorAttachments: [{
				attachment: target.colorAttachments[0].texture.createView(),
				loadValue: { r: 0.4, g: 0.2, b: 0.3, a: 1.0 },
			}],
			depthStencilAttachment: {
				attachment: target.depth.texture.createView(),
				depthLoadValue: 1.0,
				depthStoreOp: "store",
				stencilLoadValue: 0,
				stencilStoreOp: "store",
			},
			sampleCount: 1,
		};

		const commandEncoder = renderer.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

		passEncoder.setPipeline(pipeline);

		{
			let source = new ArrayBuffer(32);
			let view = new DataView(source);

			let x = 0;
			let y = 0;
			let width = 1;
			let height = 1;
			let screenWidth = size.width;
			let screenHeight = size.height;

			view.setUint32(0, 5, true);
			view.setFloat32(4, x, true);
			view.setFloat32(8, y, true);
			view.setFloat32(12, width, true);
			view.setFloat32(16, height, true);
			view.setFloat32(20, screenWidth, true);
			view.setFloat32(24, screenHeight, true);
			
			device.defaultQueue.writeBuffer(
				uniformBuffer, 0,
				source, 0, source.byteLength
			);

			passEncoder.setBindGroup(0, uniformBindGroup);
		}


		passEncoder.draw(6, 1, 0, 0);


		passEncoder.endPass();

		let commandBuffer = commandEncoder.finish();
		renderer.device.defaultQueue.submit([commandBuffer]);

	}

	frame++;

	return getTarget1(renderer).colorAttachments[0].texture;
}