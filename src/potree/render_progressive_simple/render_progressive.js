
import {Vector3, Matrix4} from "../../math/math.js";
import {RenderTarget} from "../../core/RenderTarget.js";
import {vs, fs} from "./sh_fill.js";

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
let bindGroup = null;
let uniformBuffer = null;
let target = null;

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
			{format: "r32uint"},
		],
	});

	uniformBuffer = device.createBuffer({
		size: 256,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	let size = [128, 128, 1];
	target = new RenderTarget(renderer, {
		size: size,
		colorDescriptors: [{
			size: size,
			format: renderer.swapChainFormat,
			usage: GPUTextureUsage.SAMPLED 
				| GPUTextureUsage.OUTPUT_ATTACHMENT,
		},{
			size: size,
			format: "r32uint",
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

	initialized = true;
}

function reproject(){

}

function fill(renderer, passEncoder, octree, camera){

	let {device} = renderer;
	let nodes = octree.visibleNodes;

	{ // update uniforms
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

		device.queue.writeBuffer(
			uniformBuffer, 0,
			data, data.byteOffset, data.byteLength
		);
	}

	let node = nodes[frame % nodes.length];

	let bufPosition = node.geometry.buffers.find(b => b.name === "position").buffer;
	let bufColor = node.geometry.buffers.find(b => b.name === "color").buffer;
	let vboPosition = renderer.getGpuBuffer(bufPosition);
	let vboColor = renderer.getGpuBuffer(bufColor);

	let bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: {buffer: vboColor}}
		],
	});

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	passEncoder.setVertexBuffer(0, vboPosition);

	// let numVertices = currentFillBudget;
	// let firstVertex = firstPoint;
	let numVertices = node.geometry.numElements;
	let firstVertex = 0;
	passEncoder.draw(numVertices, 1, firstVertex, 0);
	
}


export function render(renderer, octree, camera){

	init(renderer);

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
				loadValue: {r: 0.5, g: 0.5, b: 0.3, a: 1.0},
			},{
				attachment: target.colorAttachments[1].texture.createView(),
				loadValue: {r: 0, g: 0, b: 0, a: 1},
			},
		],
		depthStencilAttachment: {
			attachment: target.depth.texture.createView(),

			depthLoadValue: 1.0,
			depthStoreOp: "store",
			stencilLoadValue: 0,
			stencilStoreOp: "store",
		},
		sampleCount: 1,
	};
	Timer.timestampSep(renderer, "progressive-start");

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);


	reproject(renderer, passEncoder, octree, camera);
	fill(renderer, passEncoder, octree, camera);

	passEncoder.endPass();
	let commandBuffer = commandEncoder.finish();
	renderer.device.queue.submit([commandBuffer]);

	Timer.timestampSep(renderer, "progressive-end");
	frame++;

	return target.colorAttachments[0].texture;
}