
const vs = `
[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
	[[offset(128)]] screen_width : f32;
	[[offset(132)]] screen_height : f32;
};

[[block]] struct ColorAdjustments {
	[[offset(0)]] shift : f32;
	[[offset(4)]] scale : f32;
	[[offset(8)]] gamma : f32;
	[[offset(12)]] brightness : f32;
	[[offset(16)]] contrast : f32;
};

struct NodeBuffer{
	[[offset(0)]] color : vec4<f32>;
};

[[block]] struct Nodes{
	[[offset(0)]] values : [[stride(16)]] array<NodeBuffer, 1000>;
};

[[block]] struct U32s {
	[[offset(0)]] values : [[stride(4)]] array<u32>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), set(0)]] var<uniform> in_adjust : ColorAdjustments;
[[binding(2), set(0)]] var<uniform> nodes : Nodes;
[[binding(3), set(0)]] var<storage_buffer> ssbo_attribute : [[access(read)]]U32s;

[[location(0)]] var<in> in_position : vec4<f32>;

[[builtin(instance_index)]] var<in> instanceIdx : i32;
[[builtin(vertex_index)]] var<in> vertexID : u32;
[[builtin(position)]] var<out> out_pos : vec4<f32>;
[[location(0)]] var<out> out_color : vec4<f32>;

// formula adapted from: http://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
fn getContrastFactor(contrast : f32) -> f32{
	return (1.0158730158730156 * (contrast + 1.0)) / (1.0158730158730156 - contrast);
}

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

fn readU32(offset : u32) -> u32{
	return ssbo_attribute.values[offset];
}


import {getColor};


[[stage(vertex)]]
fn main() -> void {

	var viewPos : vec4<f32> = uniforms.worldView * in_position;
	out_pos = uniforms.proj * viewPos;

	out_color = getColor();

	return;
}
`;

const fs = `
[[location(0)]] var<in> fragColor : vec4<f32>;
[[location(0)]] var<out> outColor : vec4<f32>;

[[stage(fragment)]]
fn main() -> void {
	outColor = fragColor;

	return;
}
`;

function createColorSampler(size, numElements, type){

	if(numElements === 1){

		let reader = "";
		if(type === "uint8"){
			reader = "readU8(vertexID)";
		}else if(type === "uint16"){
			reader = "readU16(vertexID * 2u)";
		}else if(type === "uint32"){
			reader = "readU32(vertexID * 4u)";
		}

		let code = `
			fn getColor() -> vec4<f32>{

				var shift : f32 = in_adjust.shift;
				var scale : f32 = in_adjust.scale;
				var gamma : f32 = in_adjust.gamma;
				var brightness : f32 = in_adjust.brightness;
				var contrast : f32 = in_adjust.contrast;

				// shift/scale
				var value_u32 : u32 = ${reader};
				var value_f32 : f32 = (f32(value_u32) + shift) * scale;

				// gamma/brightness/contrast
				value_f32 = pow(value_f32, gamma);
				value_f32 = value_f32 + brightness;
				value_f32 = (value_f32 - 0.5) * getContrastFactor(contrast) + 0.5;

				var color : vec4<f32>;
				color.r = value_f32;
				color.g = value_f32;
				color.b = value_f32;
				color.a = 1.0;

				return color;
			}
		`;

		return code;
	}else if(numElements > 1 && size <= 4){
		throw "TODO";
	}else if(numElements >= 1 && size > 4){

		let code = `
			fn getColor() -> vec4<f32>{

				var shift : f32 = in_adjust.shift;
				var scale : f32 = in_adjust.scale;
				var gamma : f32 = in_adjust.gamma;
				var brightness : f32 = in_adjust.brightness;
				var contrast : f32 = in_adjust.contrast;

				var R : u32 = readU16(6u * vertexID + 0u);
				var G : u32 = readU16(6u * vertexID + 2u);
				var B : u32 = readU16(6u * vertexID + 4u);

				var r : f32 = f32(R);
				var g : f32 = f32(G);
				var b : f32 = f32(B);

				// shift/scale
				r = (r + shift) * scale;
				g = (g + shift) * scale;
				b = (b + shift) * scale;

				// gamma/brightness/contrast
				r = pow(r, gamma);
				r = r + brightness;
				r = (r - 0.5) * getContrastFactor(contrast) + 0.5;
				g = pow(g, gamma);
				g = g + brightness;
				g = (g - 0.5) * getContrastFactor(contrast) + 0.5;
				b = pow(b, gamma);
				b = b + brightness;
				b = (b - 0.5) * getContrastFactor(contrast) + 0.5;

				var color : vec4<f32>;
				color.r = r;
				color.g = g;
				color.b = b;
				color.a = 1.0;

				return color;
			}
		`;

		return code;
	}else{
		throw `could not generate color sampler for ${{size, numElements, type}}`;
	}



}



let cache = new Map();

export function getPipeline(renderer, octree, attributeName){
	let {device} = renderer;

	let cached = cache.get(octree);
	let needsUpdate = cached ? cached.attributeName !== attributeName : true;

	if(needsUpdate){

		let attribute = octree.loader.attributes.attributes.find(a => a.name === attributeName);

		let colorSampler = createColorSampler(attribute.byteSize, attribute.numElements, attribute.type.name);
		let vsCode = vs.replace("import {getColor};", colorSampler);

		let pipeline = device.createRenderPipeline({
			vertexStage: {
				module: device.createShaderModule({code: vsCode}),
				entryPoint: "main",
			},
			fragmentStage: {
				module: device.createShaderModule({code: fs}),
				entryPoint: "main",
			},
			primitiveTopology: "point-list",
			depthStencilState: {
				depthWriteEnabled: true,
				depthCompare: "less",
				format: "depth24plus-stencil8",
			},
			vertexState: {
				vertexBuffers: [
					{ // point position
						arrayStride: 3 * 4,
						stepMode: "vertex",
						attributes: [{ 
							shaderLocation: 0,
							offset: 0,
							format: "float3",
						}],
					}
					// ,{ // color
					// 	arrayStride: 4,
					// 	stepMode: "vertex",
					// 	attributes: [{ 
					// 		shaderLocation: 1,
					// 		offset: 0,
					// 		format: "uchar4norm",
					// 	}],
					// },
				],
			},
			rasterizationState: {
				cullMode: "none",
			},
			colorStates: [{
				format: "bgra8unorm",
			}],
		});

		cached = {pipeline, attributeName};
		cache.set(octree, cached);
	}

	return cached.pipeline;
}

