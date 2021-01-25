
import {Vector3, Matrix4} from "../../math/math.js";
import {RenderTarget} from "../../core/RenderTarget.js";

import {csDepth, precompiled as precompiled_depth, group_size as group_size_depth} from "./csDepth.js";
import {csColor, precompiled as precompiled_color, group_size as group_size_color} from "./csColor.js";
import {csReset, precompiled as precompiled_reset} from "./csReset.js";
import {vsQuad} from "./vsQuad.js";
import {fsQuad} from "./fsQuad.js";
import * as Timer from "../../renderer/Timer.js";

const FRESH_COMPILE = false;

import glslangModule from "../../../libs/glslang/glslang.js";

let glslang = null;

if(FRESH_COMPILE){
	glslangModule().then( result => {
		glslang = result;
	});
}


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

		let compiled;

		if(FRESH_COMPILE){
			compiled = glslang.compileGLSL(csDepth, "compute");
			console.log("csDepth compiled: ", compiled.join(", "));
		}else{
			compiled = precompiled_depth;
		}

		let csDescriptor = {
			code: compiled,
			source: csDepth,
		};
		let csModule = device.createShaderModule(csDescriptor);

		let uniformBufferSize = 2 * 64 + 12;
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

		let compiled;
		if(FRESH_COMPILE){
			compiled = glslang.compileGLSL(csColor, "compute");
			console.log("csColor compiled: ", compiled.join(", "));
		}else{
			compiled = precompiled_color;
		}

		let csDescriptor = {
			code: compiled,
			source: csColor,
		};
		let csModule = device.createShaderModule(csDescriptor);

		let uniformBufferSize = 2 * 64 + 12;
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

		let compiled;
		if(FRESH_COMPILE){
			compiled = glslang.compileGLSL(csReset, "compute");
			console.log("csReset compiled: ", compiled.join(", "));
		}else{
			compiled = precompiled_reset;
		}

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

		let pipeline = device.createRenderPipeline({
			vertexStage: {
				module: device.createShaderModule({code: vsQuad}),
				entryPoint: "main",
			},
			fragmentStage: {
				module: device.createShaderModule({code: fsQuad}),
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

		let uniformBufferSize = 36;
		let uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		screenPassState = {pipeline, uniformBuffer}
	}

	return screenPassState;
}

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
				resource: {buffer: uniformBuffer}
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

	let groups = Math.ceil(numUints / 128);
	passEncoder.dispatch(groups, 1, 1);
	passEncoder.endPass();

	device.defaultQueue.submit([commandEncoder.finish()]);
}

function resetBuffers(renderer){ 
	let size = renderer.getSize();
	let numUints = size.width * size.height;

	let {ssbo} = getDepthState(renderer);
	let {ssbo_colors} = getColorState(renderer);

	reset(renderer, ssbo, numUints, 0xffffff);
	reset(renderer, ssbo_colors, 4 * numUints, 0);
}

function updateUniforms(renderer, octree, camera){
	
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
		let size = renderer.getSize();

		view.setUint32(128, size.width, true);
		view.setUint32(132, size.height, true);
	}

	{ // misc
		let numPoints = octree.visibleNodes.reduce( (a, i) => a + i.geometry.numElements, 0);
		view.setUint32(136, numPoints, true);
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

let gpuBuffers = new Map();
let gpuAttributeBuffers = new Map();

function getGpuAttributeBuffer(renderer, name){

	let buffer = gpuAttributeBuffers.get(name);

	if(!buffer){

		let bpp = {
			position: 12,
			color: 4,
			rgba: 4,
		}[name];

		let size = 150_000_000 * bpp;
		let offset = 0;

		let vbo = renderer.device.createBuffer({
			size: size,
			usage: GPUBufferUsage.VERTEX 
			| GPUBufferUsage.INDEX 
			| GPUBufferUsage.STORAGE
			| GPUBufferUsage.COPY_DST,
		});

		buffer = {vbo, size, offset};

		gpuAttributeBuffers.set(name, buffer);
	}

	return buffer;
}

function getGpuBuffers(renderer, geometry){

	let buffers = gpuBuffers.get(geometry);

	if(!buffers){
		let {device} = renderer;

		let buffers = [];

		for(let entry of geometry.buffers){
			let {name, buffer} = entry;

			let gpuAttributeBuffer = getGpuAttributeBuffer(renderer, name);

			device.defaultQueue.writeBuffer(
				gpuAttributeBuffer.vbo, gpuAttributeBuffer.offset,
				buffer.buffer, 0, buffer.byteLength
			);
			gpuAttributeBuffer.offset += buffer.byteLength;

			buffers.push(gpuAttributeBuffer);
		}

		gpuBuffers.set(geometry, buffers);
	}

	return buffers;
}



function depthPass(renderer, octree, camera){
	let {device} = renderer;
	let nodes = octree.visibleNodes;

	let {pipeline, uniformBuffer} = getDepthState(renderer);
	let ssbo_depth = getDepthState(renderer).ssbo;

	const commandEncoder = device.createCommandEncoder();

	let passEncoder = commandEncoder.beginComputePass();

	Timer.timestamp(passEncoder,"depth-start");

	passEncoder.setPipeline(pipeline);

	{
		let vbo_position = gpuAttributeBuffers.get("position").vbo;
		let vbo_color = (gpuAttributeBuffers.get("color") ?? gpuAttributeBuffers.get("rgba")).vbo;

		let bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: {buffer: ssbo_depth}},
				{binding: 2, resource: {buffer: vbo_position}},
				{binding: 3, resource: {buffer: vbo_color}},
			]
		});

		passEncoder.setBindGroup(0, bindGroup);

		let numPoints = octree.visibleNodes.reduce( (a, i) => a + i.geometry.numElements, 0);
		let groups = Math.ceil(numPoints / group_size_depth);
		passEncoder.dispatch(groups);
	}

	Timer.timestamp(passEncoder,"depth-end");

	passEncoder.endPass();

	Timer.resolve(renderer, commandEncoder);

	device.defaultQueue.submit([commandEncoder.finish()]);

}

function colorPass(renderer, octree, camera){ 
	let {device} = renderer;
	let nodes = octree.visibleNodes;

	let {pipeline, uniformBuffer, ssboSize} = getColorState(renderer);
	let {ssbo_colors} = getColorState(renderer);
	let ssbo_depth = getDepthState(renderer).ssbo;

	const commandEncoder = device.createCommandEncoder();
	let passEncoder = commandEncoder.beginComputePass();

	Timer.timestamp(passEncoder,"color-start");

	passEncoder.setPipeline(pipeline);

	{
		let vbo_position = gpuAttributeBuffers.get("position").vbo;
		let vbo_color = (gpuAttributeBuffers.get("color") ?? gpuAttributeBuffers.get("rgba")).vbo;

		let bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: {buffer: ssbo_colors}},
				{binding: 2, resource: {buffer: ssbo_depth}},
				{binding: 3, resource: {buffer: vbo_position}},
				{binding: 4, resource: {buffer: vbo_color}},
			]
		});

		passEncoder.setBindGroup(0, bindGroup);

		let numPoints = octree.visibleNodes.reduce( (a, i) => a + i.geometry.numElements, 0);
		let groups = Math.ceil(numPoints / group_size_color);
		passEncoder.dispatch(groups, 1, 1);
	}

	Timer.timestamp(passEncoder,"color-end");

	passEncoder.endPass();
	Timer.resolve(renderer, commandEncoder);
	device.defaultQueue.submit([commandEncoder.finish()]);

}

function resolve(renderer, octree, camera){
	let {device} = renderer;

	let ssbo_depth = getDepthState(renderer).ssbo;
	let {ssbo_colors} = getColorState(renderer);
	let {pipeline, uniformBuffer} = getScreenPassState(renderer);
	let target = getTarget1(renderer);
	let size = renderer.getSize();

	let uniformBindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: {buffer: ssbo_colors}},
			{binding: 2, resource: {buffer: ssbo_depth}}
		],
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

	Timer.timestamp(passEncoder,"resolve-start");

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
		let pixelWindow = Math.floor(guiContent["point size"] / 2);

		view.setUint32(0, 5, true);
		view.setFloat32(4, x, true);
		view.setFloat32(8, y, true);
		view.setFloat32(12, width, true);
		view.setFloat32(16, height, true);
		view.setFloat32(20, screenWidth, true);
		view.setFloat32(24, screenHeight, true);
		view.setUint32(28, pixelWindow, true);
		
		device.defaultQueue.writeBuffer(
			uniformBuffer, 0,
			source, 0, source.byteLength
		);

		passEncoder.setBindGroup(0, uniformBindGroup);
	}

	passEncoder.draw(6, 1, 0, 0);

	Timer.timestamp(passEncoder,"resolve-end");

	passEncoder.endPass();

	Timer.resolve(renderer, commandEncoder);

	let commandBuffer = commandEncoder.finish();
	renderer.device.defaultQueue.submit([commandBuffer]);

}

export function render(renderer, octree, camera){

	if(FRESH_COMPILE && !glslang){
		console.log("glslang not yet initialized");

		return;
	}

	let target = getTarget1(renderer);

	if(octree.visibleNodes.length === 0){
		return target.colorAttachments[0].texture;
	}

	{ // RESIZE RENDER TARGET
		let size = renderer.getSize();
		target.setSize(size.width, size.height);
	}

	resetBuffers(renderer);
	updateUniforms(renderer, octree, camera);

	for(let node of octree.visibleNodes){
		getGpuBuffers(renderer, node.geometry);
	}

	depthPass(renderer, octree, camera);
	colorPass(renderer, octree, camera);
	resolve(renderer, octree, camera);

	return target.colorAttachments[0].texture;
}