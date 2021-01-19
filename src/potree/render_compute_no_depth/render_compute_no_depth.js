
import {Vector3, Matrix4} from "../../math/math.js";
import {RenderTarget} from "../../core/RenderTarget.js";

import {csColor, group_size as group_size_color} from "./csColor.js";
import {csReset} from "./csReset.js";
import {vsQuad} from "./vsQuad.js";
import {fsQuad} from "./fsQuad.js";

const FRESH_COMPILE = true;

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

let colorState = null;
let resetState = null;
let screenPassState = null;

function getColorState(renderer){

	if(!colorState){

		let {device} = renderer;

		let ssboSize = 4 * 2560 * 1440 * 4;
		let ssbo_colors = renderer.createBuffer(ssboSize);

		let compiled;
		if(FRESH_COMPILE){
			compiled = glslang.compileGLSL(csColor, "compute");
			console.log("csColor compiled: ", compiled.join(", "));
		}else{

			compiled = color_precompiled;
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

			compiled = new Uint32Array([
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

		let uniformBufferSize = 32;
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

	let {ssbo_colors} = getColorState(renderer);

	reset(renderer, ssbo_colors, numUints, 0);
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

	{ // set color pass uniforms
		let {uniformBuffer} = getColorState(renderer);
		device.defaultQueue.writeBuffer(
			uniformBuffer, 0,
			data, data.byteOffset, data.byteLength
		);
	}
}

let bindGroupCache_color = new Map();

function colorPass(renderer, octree, camera){ 
	let {device} = renderer;
	let nodes = octree.visibleNodes;

	let {pipeline, uniformBuffer, ssboSize} = getColorState(renderer);
	let {ssbo_colors} = getColorState(renderer);

	const commandEncoder = device.createCommandEncoder();
	let passEncoder = commandEncoder.beginComputePass();

	passEncoder.setPipeline(pipeline);

	for(let node of nodes){

		let gpuBuffers = renderer.getGpuBuffers(node.geometry);

		let bindGroup = bindGroupCache_color.get(node);

		if(!bindGroup){
			bindGroup = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: [
					{binding: 0, resource: {buffer: uniformBuffer}},
					{binding: 1, resource: {buffer: ssbo_colors}},
					{binding: 2, resource: {buffer: gpuBuffers[0].vbo}},
					{binding: 3, resource: {buffer: gpuBuffers[1].vbo}}
				]
			});

			bindGroupCache_color.set(node, bindGroup);
		}

		passEncoder.setBindGroup(0, bindGroup);

		let groups = Math.ceil(node.geometry.numElements / group_size_color);
		passEncoder.dispatch(groups, 1, 1);
	}

	passEncoder.endPass();
	device.defaultQueue.submit([commandEncoder.finish()]);

}

function resolve(renderer, octree, camera){
	let {device} = renderer;

	let {ssbo_colors} = getColorState(renderer);
	let {pipeline, uniformBuffer} = getScreenPassState(renderer);
	let target = getTarget1(renderer);
	let size = renderer.getSize();

	let uniformBindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: {buffer: ssbo_colors}},
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

export function renderComputeNoDepth(renderer, octree, camera){

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

	colorPass(renderer, octree, camera);
	resolve(renderer, octree, camera);

	return target.colorAttachments[0].texture;
}