

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
}
`;

const getColor_scalar = `
fn getColor(vertex : VertexInput) -> vec4<f32> {

	var w : f32 = f32(vertex.attribute) / 256.0;

	var color : vec4<f32> = vec4<f32>(
		w,
		w,
		w,
		1.0
	);

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
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

struct VertexInput {
	[[builtin(instance_index)]] instanceIdx : u32;
	[[builtin(vertex_index)]] vertexID : u32;
	[[location(0)]] position : vec4<f32>;
	${strAttribute}
};

struct VertexOutput {
	[[builtin(position)]] position : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
};

${fnGetColor}

[[stage(vertex)]]
fn main(vertex : VertexInput) -> VertexOutput {

	var output : VertexOutput;

	var viewPos : vec4<f32> = uniforms.worldView * vertex.position;
	output.position = uniforms.proj * viewPos;

	output.color = getColor(vertex);

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




