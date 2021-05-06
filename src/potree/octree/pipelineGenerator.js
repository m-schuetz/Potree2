
import {generate as generateShaders} from "./shaderGenerator.js";

function getStride(attribute){
	return 4;
}

function getFormat(attribute){
	let typename = attribute.type.name
	let numElements = attribute.numElements;

	if(typename === "uint16" && numElements === 1){
		return "uint32";
	}else if(typename === "uint16" && numElements === 3){
		return "unorm8x4";
	}else{
		throw "unsupported type";
	}
}

export function generate(renderer, args = {}){

	let {device} = renderer;
	let {vs, fs} = generateShaders(args);

	let attributeBufferDescriptor = { 
		arrayStride: getStride(args.attribute),
		stepMode: "vertex",
		attributes: [{ 
			shaderLocation: 1,
			offset: 0,
			format: getFormat(args.attribute),
		}],
	};

	const pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
			buffers: [
				{ // point position
					arrayStride: 3 * 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				},
				attributeBufferDescriptor
			],
		},
		fragment: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'point-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: "greater",
			format: "depth32float",
		},
	});

	return pipeline;
}