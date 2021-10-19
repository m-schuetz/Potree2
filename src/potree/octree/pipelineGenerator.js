
import {generate as generateShaders} from "./shaderGenerator.js";

export function generate(renderer, args = {}){

	let {device} = renderer;
	let {shaderSource} = generateShaders(args);
	let {flags} = args;

	let depthWrite = true;
	let blend = {
		color: {
			srcFactor: "one",
			dstFactor: "zero",
			operation: "add",
		},
		alpha: {
			srcFactor: "one",
			dstFactor: "zero",
			operation: "add",
		},
	};

	let isAdditive = flags.includes("additive_blending");
	let format = "bgra8unorm";

	// isAdditive = true;
	if(isAdditive){
		format = "rgba16float";
		depthWrite = false;

		blend = {
			color: {
				srcFactor: "one",
				dstFactor: "one",
				operation: "add",
			},
			alpha: {
				srcFactor: "one",
				dstFactor: "one",
				operation: "add",
			},
		};
	}
	
	console.groupCollapsed("compiling octree shader");
	console.log("==== SHADER ====");
	console.log(shaderSource);
	console.groupEnd();

	let module = device.createShaderModule({code: shaderSource});

	const layout_0 = renderer.device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'uniform'},
			},{
				binding: 1,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			}
		],
	});

	const layout_1 = renderer.device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				sampler: {sampleType: 'filtering'},
			},{
				binding: 1,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				texture: {sampleType : 'float'},
			}
		],
	});

	const layout_2 = renderer.device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			}
		],
	});

	const layout_3 = renderer.device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			}
		],
	});

	const pipeline = device.createRenderPipeline({
		layout: device.createPipelineLayout({
			bindGroupLayouts: [
				layout_0,
				layout_1,
				layout_2,
				layout_3,
			],
		}),
		vertex: {
			module: module,
			entryPoint: "main_vertex",
			buffers: [],
		},
		fragment: {
			module: module,
			entryPoint: "main_fragment",
			targets: [{format: format, blend: blend}],
		},
		primitive: {
			topology: 'point-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: depthWrite,
			depthCompare: "greater",
			format: "depth32float",
		},
	});

	return pipeline;
}