

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

function createVS(args){

	let type = toWgslType(args.attribute);
	let strAttribute = `[[location(${1})]] attribute : ${type};\n`;


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

struct AttributeDescriptor{
	offset      : u32;
	numElements : u32;
	valuetype   : u32;
	range_min   : f32;
	range_max   : f32;
	clamp       : u32;
};

[[block]] struct Nodes{
	values : [[stride(8)]] array<Node>;
};

[[block]] struct AttributeDescriptors{
	values : [[stride(24)]] array<AttributeDescriptor>;
};

[[block]] struct U32s {
	values : [[stride(4)]] array<u32>;
};

let TYPES_U8            =  0u;
let TYPES_U16           =  1u;
let TYPES_U32           =  2u;
let TYPES_I8            =  3u;
let TYPES_I16           =  4u;
let TYPES_I32           =  5u;
let TYPES_F32           =  6u;
let TYPES_F64           =  7u;
let TYPES_RGBA          = 50u;
let TYPES_ELEVATION     = 51u;

let CLAMP_DISABLED      =  0u;
let CLAMP_ENABLED       =  1u;

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), group(0)]] var<storage, read> attributes : AttributeDescriptors;

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

fn readU8(offset : u32) -> u32{
	var ipos : u32 = offset / 4u;
	var val_u32 : u32 = buffer.values[ipos];

	// var shift : u32 = 8u * (3u - (offset % 4u));
	var shift : u32 = 8u * (offset % 4u);
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


fn scalarToColor(vertex : VertexInput, attribute : AttributeDescriptor, node : Node) -> vec4<f32> {

	var value : f32 = 0.0;

	if(attribute.valuetype == TYPES_U8){
		var offset = node.numPoints * attribute.offset + 1u * vertex.vertexID;
		value = f32(readU8(offset));
	}elseif(attribute.valuetype == TYPES_F64){
		var offset = node.numPoints * attribute.offset + 8u * vertex.vertexID;

		var b0 = readU8(offset + 0u);
		var b1 = readU8(offset + 1u);
		var b2 = readU8(offset + 2u);
		var b3 = readU8(offset + 3u);
		var b4 = readU8(offset + 4u);
		var b5 = readU8(offset + 5u);
		var b6 = readU8(offset + 6u);
		var b7 = readU8(offset + 7u);

		var exponent_f64_bin = (b7 << 4u) | (b6 >> 4u);
		var exponent_f64 = exponent_f64_bin - 1023u;

		var exponent_f32 = exponent_f64 + 127u;
		var mantissa_f32 = (b6 & 0x0Fu) << 19u
			| b5 << 11u
			| b4 << 3u;
		var sign = b7 >> 7u;
		var value_u32 = sign << 31u
			| exponent_f32 << 23u
			| mantissa_f32;

		var value_f32 = bitcast<f32>(value_u32);

		value = value_f32;
	}elseif(attribute.valuetype == TYPES_ELEVATION){
		value = vertex.position.z;
	}elseif(attribute.valuetype == TYPES_U16){
		var offset = node.numPoints * attribute.offset + 2u * vertex.vertexID;
		value = f32(readU16(offset));
	}

	var w = (value - attribute.range_min) / (attribute.range_max - attribute.range_min);

	if(attribute.clamp == CLAMP_ENABLED){
		w = clamp(w, 0.0, 1.0);
	}

	var uv : vec2<f32> = vec2<f32>(w, 0.0);
	var color = textureSampleLevel(myTexture, mySampler, uv, 0.0);

	return color;
}

fn vectorToColor(vertex : VertexInput, attribute : AttributeDescriptor, node : Node) -> vec4<f32> {

	var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

	if(attribute.valuetype == TYPES_RGBA){
		color = vertex.attribute;
	}

	return color;
}


[[stage(vertex)]]
fn main(vertex : VertexInput) -> VertexOutput {

	{ // reference all potentially unused variables, 
		// otherwise the bind group layout breaks if they're not used in the shader
		ignore(mySampler);
		ignore(myTexture);

		var dbg = buffer.values[0];
		var dbg1 = attributes.values[0];
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

	{ // COLORIZE BY ATTRIBUTE DESCRIPTORS
		var task = attributes.values[0];
		var value : f32 = 0.0;

		var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

		if(task.numElements == 1u){
			color = scalarToColor(vertex, task, node);
		}else{
			color = vectorToColor(vertex, task, node);
		}

		output.color = color;
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




