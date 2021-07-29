

function toWgslType(attribute){

	let typename = attribute.type.name
	let numElements = attribute.numElements;

	if(typename === "uint16" && numElements === 1){
		return "u32";
	}else if(typename === "uint16" && numElements === 3){
		return "vec4<f32>";
	}else{
		throw "unsupported type";
	}

}

const getColor_rgba = `
fn getColor(vertex : VertexInput) -> vec4<f32> {
	return vertex.attribute;

	// var uv : vec2<f32> = vec2<f32>(vertex.position.z * 0.2, 0.0);
	// var color : vec4<f32> = textureSampleLevel(myTexture, mySampler, uv, 0.0);

	// return color;
}
`;

const getColor_elevation = `
fn getColor_elevation(vertex : VertexInput) -> vec4<f32> {

	var min = 3.0;
	var max = 15.0;
	var size = max - min;

	var w = (vertex.position.z - min) / size;
	// w = -clamp(-w, 0.0, 1.0);

	var uv : vec2<f32> = vec2<f32>(w, 0.0);
	var color : vec4<f32> = textureSampleLevel(myTexture, mySampler, uv, 0.0);

	return color;
}
`;

const getColor_scalar = `
fn getColor(vertex : VertexInput) -> vec4<f32> {

	var w : f32 = (f32(vertex.attribute) - 0.0) / 255.0;
	w = clamp(1.0 - w, 0.0, 1.0);

	// var w : f32 = f32(vertex.attribute % 10u) / 10.0;
	// w = clamp(1.0 - w, 0.0, 1.0);

	var uv : vec2<f32> = vec2<f32>(w, 0.0);
	var color : vec4<f32> = textureSampleLevel(myTexture, mySampler, uv, 0.0);

	return color;
}
`;


function createVS(args){

	let type = toWgslType(args.attribute);
	let strAttribute = `[[location(${1})]] attribute : ${type};\n`;
	
	let fnGetColor;

	if(args.mapping === "rgba"){
		fnGetColor = getColor_rgba;
	}else if(args.mapping === "scalar"){
		fnGetColor = getColor_scalar;
	}


let shader = `
[[block]] struct Uniforms {
	[[size(64)]] worldView : mat4x4<f32>;
	[[size(64)]] proj : mat4x4<f32>;
	[[size(4)]] screen_width : f32;
	[[size(4)]] screen_height : f32;
	[[size(4)]] hqs_flag : u32;
	[[size(4)]] colorMode : u32;
};

struct Node {
	[[size(4)]] numPoints : u32;
	[[size(4)]] dbg : u32;
};

[[block]] struct Nodes{
	values : [[stride(8)]] array<Node>;
};

[[block]] struct U32s {
	values : [[stride(4)]] array<u32>;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(0), group(1)]] var mySampler: sampler;

[[binding(1), group(1)]] var myTexture: texture_2d<f32>;

[[binding(0), group(2)]] var<storage, read> buffer : U32s;
[[binding(0), group(3)]] var<storage, read> nodes : Nodes;

struct VertexInput {
	[[builtin(instance_index)]] instanceID : u32;
	[[builtin(vertex_index)]] vertexID : u32;
	[[location(0)]] position : vec4<f32>;
	${strAttribute}
};

struct VertexOutput {
	[[builtin(position)]] position : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
};

let COLOR_MODE_RGBA        = 0u;
let COLOR_MODE_ELEVATION   = 1u;

${fnGetColor}
${getColor_elevation}

fn readU8(offset : u32) -> u32{
	var ipos : u32 = offset / 4u;
	var val_u32 : u32 = buffer.values[ipos];

	var shift : u32 = 8u * (3u - (offset % 4u));
	var val_u8 : u32 = (val_u32 >> shift) & 0xFFu;

	return val_u8;
}

fn readU16(offset : u32) -> u32{
	
	var first = readU8(offset + 0u);
	var second = readU8(offset + 1u);

	var value = first | (second << 8u);

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


[[stage(vertex)]]
fn main(vertex : VertexInput) -> VertexOutput {

	{ // reference all potentially unused variables, 
		// otherwise the bind group layout breaks if they're not used in the shader
		ignore(mySampler);
		ignore(myTexture);

		var dbg = buffer.values[0];
	}

	var node = nodes.values[vertex.instanceID];
	var viewPos : vec4<f32> = uniforms.worldView * vertex.position;

	var output : VertexOutput;
	output.position = uniforms.proj * viewPos;

	// in the HQS depth pass, shift points 1% further away from camera
	if(uniforms.hqs_flag > 0u){
		output.position = output.position / output.position.w;

		viewPos.z = viewPos.z * 1.01;
		
		var shifted : vec4<f32> = uniforms.proj * viewPos;
		output.position.z = shifted.z / shifted.w;
	}

	// { // INTENSITY
	// 	var offset = node.numPoints * 16u + 2u * vertex.vertexID;
	// 	var value = readU16(offset);

	// 	var w = f32(value) / 60000.0;
	// 	output.color = vec4<f32>(w, w, w, 1.0);
	// }

	// { // RETURN NUMBER
	// 	var offset = node.numPoints * 18u + vertex.vertexID;
	// 	var value = readU8(offset);

	// 	var w = f32(value) / 4.0;
	// 	output.color = vec4<f32>(w, w, w, 1.0);
	// }

	{ // NUMBER OF RETURNS
		var offset = node.numPoints * 19u + vertex.vertexID;
		var value = readU8(offset);

		var w = f32(value) / 4.0;
		output.color = vec4<f32>(w, w, w, 1.0);
	}

	// { // POINT SOURCE ID
	// 	var intensityOffset = node.numPoints * 23u + 2u * vertex.vertexID;
	// 	var intensity = readU16(intensityOffset);

	// 	var w = f32(intensity) / 10.0;
	// 	output.color = vec4<f32>(w, w, w, 1.0);

	// 	var uv : vec2<f32> = vec2<f32>(w, 0.0);
	// 	var color : vec4<f32> = textureSampleLevel(myTexture, mySampler, uv, 0.0);

	// 	output.color = color;
	// }

	{ // ELEVATION
		var min = 3.0;
		var max = 150.0;
		var size = max - min;

		var w = (vertex.position.z - min) / size;

		var uv : vec2<f32> = vec2<f32>(w, 0.0);
		var color : vec4<f32> = textureSampleLevel(myTexture, mySampler, uv, 0.0);

		output.color = color;
	}

	{ // GPS-TIME (25)
		var offset = node.numPoints * 25u + 8u * vertex.vertexID;

		var b0 = readU8(offset + 0u);
		var b1 = readU8(offset + 1u);
		var b2 = readU8(offset + 2u);
		var b3 = readU8(offset + 3u);
		var b4 = readU8(offset + 4u);
		var b5 = readU8(offset + 5u);
		var b6 = readU8(offset + 6u);
		var b7 = readU8(offset + 7u);

		

	}


	return output;
}
`;

	return shader;
}

const fsBase = `

struct FragmentInput {
	[[location(0)]] color : vec4<f32>;
};

struct FragmentOutput {
	[[location(0)]] color : vec4<f32>;
};

[[stage(fragment)]]
fn main(fragment : FragmentInput) -> FragmentOutput {
	var output : FragmentOutput;
	output.color = fragment.color;

	return output;
}
`;


export function generate(args = {}){

	let vs = createVS(args);
	let fs = fsBase;

	return {vs, fs};
}




