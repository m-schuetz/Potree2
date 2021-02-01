
import {Vector3, Matrix4} from "../../math/math.js";
import {RenderTarget} from "../../core/RenderTarget.js";
import {vs, fs} from "./sh_fill.js";
import * as sh_reproject from "./sh_reproject.js";

import * as Timer from "../../renderer/Timer.js";


let firstPoint = 0;
let fillBudget = 1_000_000;
let currentFillBudget = fillBudget;
let makeStep = false;
let frame = 0;

function step(octree){

	// if(!makeStep && frame > 200){
	// 	return;
	// }else{
	// 	makeStep = false;
	// }

	let numPoints = octree.visibleNodes.reduce( (a, i) => a + i.geometry.numElements, 0);
	firstPoint = firstPoint + fillBudget;
	if(firstPoint > numPoints){
		firstPoint = 0;
	}
	currentFillBudget = Math.min(numPoints - firstPoint, fillBudget);
}


let initialized = false;
let pipeline = null;
let pipeline_reproject;
let bindGroup = null;
let uniformBuffer = null;
let sampler = null;

let targets = [];
let target_dilate = null;

function init(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	pipeline = device.createRenderPipeline({
		vertexStage: {
			module: device.createShaderModule({code: vs}),
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
			],
		},
		rasterizationState: {
			cullMode: "none",
		},
		colorStates: [
			{format: "bgra8unorm"},
			{format: "rgba32float"},
		],
	});

	pipeline_reproject = device.createRenderPipeline({
		vertexStage: {
			module: device.createShaderModule({code: sh_reproject.vs}),
			entryPoint: "main",
		},
		fragmentStage: {
			module: device.createShaderModule({code: sh_reproject.fs}),
			entryPoint: "main",
		},
		primitiveTopology: "point-list",
		depthStencilState: {
			depthWriteEnabled: true,
			depthCompare: "less",
			format: "depth24plus-stencil8",
		},
		vertexState: {
			vertexBuffers: [],
		},
		rasterizationState: {
			cullMode: "none",
		},
		colorStates: [
			{format: "bgra8unorm"},
			{format: "rgba32float"},
		],
	});

	pipeline_dilate = device.createRenderPipeline({
		vertexStage: {
			module: device.createShaderModule({code: sh_dilate.vs}),
			entryPoint: "main",
		},
		fragmentStage: {
			module: device.createShaderModule({code: sh_dilate.fs}),
			entryPoint: "main",
		},
		primitiveTopology: "point-list",
		depthStencilState: {
			depthWriteEnabled: true,
			depthCompare: "less",
			format: "depth24plus-stencil8",
		},
		vertexState: {
			vertexBuffers: [],
		},
		rasterizationState: {
			cullMode: "none",
		},
		colorStates: [
			{format: "bgra8unorm"},
		],
	});

	sampler = device.createSampler({
		magFilter: 'nearest',
		minFilter: 'nearest',
		mipmapFilter : 'nearest',
		addressModeU: "repeat",
		addressModeV: "repeat",
		maxAnisotropy: 1,
	});

	uniformBuffer = device.createBuffer({
		size: 256,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	let size = [128, 128, 1];

	let target1 = new RenderTarget(renderer, {
		size: size,
		colorDescriptors: [{
			size: size,
			format: renderer.swapChainFormat,
			usage: GPUTextureUsage.SAMPLED 
				| GPUTextureUsage.OUTPUT_ATTACHMENT,
		},{
			size: size,
			format: "rgba32float",
			usage: GPUTextureUsage.SAMPLED 
				| GPUTextureUsage.OUTPUT_ATTACHMENT,
		}],
		depthDescriptor: {
			size: size,
			format: "depth24plus-stencil8",
			usage: GPUTextureUsage.SAMPLED 
				| GPUTextureUsage.OUTPUT_ATTACHMENT,
		}
	});

	let target2 = new RenderTarget(renderer, {
		size: size,
		colorDescriptors: [{
			size: size,
			format: renderer.swapChainFormat,
			usage: GPUTextureUsage.SAMPLED 
				| GPUTextureUsage.OUTPUT_ATTACHMENT,
		},{
			size: size,
			format: "rgba32float",
			usage: GPUTextureUsage.SAMPLED 
				| GPUTextureUsage.OUTPUT_ATTACHMENT,
		}],
		depthDescriptor: {
			size: size,
			format: "depth24plus-stencil8",
			usage: GPUTextureUsage.SAMPLED 
				| GPUTextureUsage.OUTPUT_ATTACHMENT,
		}
	});

	target_dilate = new RenderTarget(renderer, {
		size: size,
		colorDescriptors: [{
			size: size,
			format: renderer.swapChainFormat,
			usage: GPUTextureUsage.SAMPLED 
				| GPUTextureUsage.OUTPUT_ATTACHMENT,
		}],
		depthDescriptor: {
			size: size,
			format: "depth24plus-stencil8",
			usage: GPUTextureUsage.SAMPLED 
				| GPUTextureUsage.OUTPUT_ATTACHMENT,
		}
	});

	targets = [target1, target2];

	initialized = true;
}

function updateUniforms(renderer, octree){

	let {device} = renderer;

	let data = new ArrayBuffer(140);
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
		let size = targets[frame % 2].size;

		view.setFloat32(128, size[0], true);
		view.setFloat32(132, size[1], true);
	}

	device.queue.writeBuffer(
		uniformBuffer, 0,
		data, data.byteOffset, data.byteLength
	);

}

function reproject(renderer, octree, camera){

	let {device} = renderer;

	let target_prev = targets[Math.abs((frame - 1) % 2)];
	let target_curr = targets[frame % 2];

	let renderPassDescriptor = {
		colorAttachments: [
			{
				attachment: target_curr.colorAttachments[0].texture.createView(),
				loadValue: {r: 0.4, g: 0.1, b: 0.2, a: 1.0},
			},{
				attachment: target_curr.colorAttachments[1].texture.createView(),
				loadValue: {r: 0, g: 0, b: 0, a: 1},
			},
		],
		depthStencilAttachment: {
			attachment: target_curr.depth.texture.createView(),

			depthLoadValue: 1.0,
			depthStoreOp: "store",
			stencilLoadValue: 0,
			stencilStoreOp: "store",
		},
		sampleCount: 1,
	};

	let bindGroup = device.createBindGroup({
		layout: pipeline_reproject.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: sampler},
			{binding: 2, resource: target_prev.colorAttachments[0].texture.createView()},
			{binding: 3, resource: target_prev.colorAttachments[1].texture.createView()},
		],
	});

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

	passEncoder.setPipeline(pipeline_reproject);
	passEncoder.setBindGroup(0, bindGroup);
	
	let numPixels = target_prev.size[0] * target_prev.size[1];
	passEncoder.draw(numPixels, 1, 0, 0);

	passEncoder.endPass();
	let commandBuffer = commandEncoder.finish();
	renderer.device.queue.submit([commandBuffer]);


}


let bindGroupCache = new Map();

function fill(renderer, passEncoder, octree, camera){

	let {device} = renderer;
	let nodes = octree.visibleNodes;

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

	passEncoder.setPipeline(pipeline);

	// for(let node of nodes){
	//	let node = nodes[frame % nodes.length];
	let n = 20;
	let start = (n * frame) % nodes.length;
	for(let i = start; i < start + n; i++){

		let node = nodes[i % nodes.length];

		let bufPosition = node.geometry.buffers.find(b => b.name === "position").buffer;
		let bufColor = node.geometry.buffers.find(b => b.name === "rgba").buffer;
		let vboPosition = renderer.getGpuBuffer(bufPosition);
		let vboColor = renderer.getGpuBuffer(bufColor);

		let bindGroup = bindGroupCache.get(node);
		if(!bindGroup){
			bindGroup = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: [
					{binding: 0, resource: {buffer: uniformBuffer}},
					{binding: 1, resource: {buffer: vboColor}}
				],
			});

			bindGroupCache.set(node, bindGroup);
		}
		

		passEncoder.setBindGroup(0, bindGroup);
		passEncoder.setVertexBuffer(0, vboPosition);

		// let bins = 2;
		// let bin = frame % bins;
		// let firstVertex = Math.floor(bin * (node.geometry.numElements / bins));
		// let numVertices = Math.floor(node.geometry.numElements / bins);

		let firstVertex = 0;
		let numVertices = node.geometry.numElements;

		passEncoder.draw(numVertices, 1, firstVertex, 0);
	}

	passEncoder.endPass();
	let commandBuffer = commandEncoder.finish();
	renderer.device.queue.submit([commandBuffer]);
}


export function render(renderer, octree, camera){

	init(renderer);

	let target = targets[frame % 2];

	if(octree.visibleNodes.length === 0){
		return target.colorAttachments[0].texture;
	}

	{ // RESIZE RENDER TARGET
		let size = renderer.getSize();
		target.setSize(size.width, size.height);
	}

	step(octree);


	let renderPassDescriptor = {
		colorAttachments: [
			{
				attachment: target.colorAttachments[0].texture.createView(),
				loadValue: "load",
			},{
				attachment: target.colorAttachments[1].texture.createView(),
				loadValue: "load",
			},
		],
		depthStencilAttachment: {
			attachment: target.depth.texture.createView(),

			depthLoadValue: "load",
			depthStoreOp: "store",
			stencilLoadValue: 0,
			stencilStoreOp: "store",
		},
		sampleCount: 1,
	};
	Timer.timestampSep(renderer, "progressive-start");


	updateUniforms(renderer, octree);

	reproject(renderer, octree, camera);
	fill(renderer, octree, camera);
	dilate(renderer, octree, camera);

	Timer.timestampSep(renderer, "progressive-end");
	frame++;

	return target.colorAttachments[0].texture;
}