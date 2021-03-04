
import {Vector3, Matrix4} from "../../math/math.js";
import {RenderTarget} from "../../core/RenderTarget.js";

import {csDepth, precompiled as precompiled_depth, group_size as group_size_depth} from "./csDepth.js";
import {csColor, precompiled as precompiled_color, group_size as group_size_color} from "./csColor.js";
import {csReset, precompiled as precompiled_reset} from "./csReset.js";
import {csPrepare} from "./csPrepare.js";
import {csReproject} from "./csReproject.js";
import {vsQuad} from "./vsQuad.js";
import {fsQuad} from "./fsQuad.js";
import * as Timer from "../../renderer/Timer.js";

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
					| GPUTextureUsage.RENDER_ATTACHMENT,
			}],
			depthDescriptor: {
				size: size,
				format: "depth24plus-stencil8",
				usage: GPUTextureUsage.SAMPLED 
					// | GPUTextureUsage.COPY_SRC 
					// | GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.RENDER_ATTACHMENT,
			}
		});
	}

	return _target_1;
}


let depthState = null;
let colorState = null;
let resetState = null;
let screenPassState = null;
let progressiveState = null;

function getProgressiveState(renderer){

	if(!progressiveState){
		let {device} = renderer;
		let maxPoints = 2560 * 1440;
		let bytesPerPoint = 12 + 4 + 4; // pos, color, pointID

		let ssbo_prep_meta = device.createBuffer({
			size: 256,
			usage: GPUBufferUsage.STORAGE
				| GPUBufferUsage.COPY_SRC
				| GPUBufferUsage.COPY_DST
				| GPUBufferUsage.INDIRECT,
		}); 
		// let ssbo_prep_pos = renderer.createBuffer(maxPoints * 12);
		// let ssbo_prep_col = renderer.createBuffer(maxPoints * 4);
		let ssbo_prep = renderer.createBuffer(maxPoints * bytesPerPoint);
		let ssbo_point_id = renderer.createBuffer(maxPoints * 4);

		
		let pipeline_prepare;
		{ // PREPARE
			let compiled = glslang.compileGLSL(csPrepare, "compute");
			let csDescriptor = {
				code: compiled,
				source: csPrepare,
			};
			let csModule = device.createShaderModule(csDescriptor);

			pipeline_prepare = device.createComputePipeline({
				computeStage: {
					module: csModule,
					entryPoint: "main",
				}
			});
		}

		let pipeline_reproject;
		{ // REPROJECT
			let compiled = glslang.compileGLSL(csReproject, "compute");
			let csDescriptor = {
				code: compiled,
				source: csReproject,
			};
			let csModule = device.createShaderModule(csDescriptor);

			pipeline_reproject = device.createComputePipeline({
				computeStage: {
					module: csModule,
					entryPoint: "main",
				}
			});
		}

		let uniformBufferSize = 256;
		let uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});


		progressiveState = {
			pipeline_reproject, pipeline_prepare, uniformBuffer,
			ssbo_prep_meta, ssbo_prep, ssbo_point_id
		};
	}

	return progressiveState;
}

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

		let uniformBufferSize = 2 * 64 + 16;
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

		let uniformBufferSize = 2 * 64 + 16;
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
		device.queue.writeBuffer(
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

	device.queue.submit([commandEncoder.finish()]);
}

function resetBuffers(renderer){ 
	let size = renderer.getSize();
	let numUints = size.width * size.height;

	let {ssbo} = getDepthState(renderer);
	let {ssbo_colors} = getColorState(renderer);
	let {ssbo_point_id, ssbo_prep_meta} = getProgressiveState(renderer);

	reset(renderer, ssbo, numUints, 0xffffff);
	reset(renderer, ssbo_colors, 4 * numUints, 0);
	reset(renderer, ssbo_point_id, numUints, 0);
	reset(renderer, ssbo_prep_meta, 1, 0);
}

function updateUniforms(renderer, octree, camera){
	
	let {device} = renderer;

	let data = new ArrayBuffer(144);
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
		//let numPoints = octree.visibleNodes.reduce( (a, i) => a + i.geometry.numElements, 0);

		view.setUint32(136, firstPoint, true);
		view.setUint32(140, currentFillBudget, true);
	}

	{ // set depth pass uniforms
		let {uniformBuffer} = getDepthState(renderer);
		device.queue.writeBuffer(
			uniformBuffer, 0,
			data, data.byteOffset, data.byteLength
		);
	}

	{ // set color pass uniforms
		let {uniformBuffer} = getColorState(renderer);
		device.queue.writeBuffer(
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

		let size = 100_000_000 * bpp;
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

			device.queue.writeBuffer(
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
		let groups = Math.ceil(currentFillBudget / group_size_depth);
		passEncoder.dispatch(groups);
	}

	Timer.timestamp(passEncoder,"depth-end");

	passEncoder.endPass();

	Timer.resolve(renderer, commandEncoder);

	device.queue.submit([commandEncoder.finish()]);

}

function colorPass(renderer, octree, camera){ 
	let {device} = renderer;
	let nodes = octree.visibleNodes;

	let {pipeline, uniformBuffer} = getColorState(renderer);
	let {ssbo_colors} = getColorState(renderer);
	let ssbo_depth = getDepthState(renderer).ssbo;
	let {ssbo_point_id} = getProgressiveState(renderer);

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
				{binding: 5, resource: {buffer: ssbo_point_id}},
			]
		});

		passEncoder.setBindGroup(0, bindGroup);

		let numPoints = octree.visibleNodes.reduce( (a, i) => a + i.geometry.numElements, 0);
		let groups = Math.ceil(currentFillBudget / group_size_color);
		passEncoder.dispatch(groups, 1, 1);
	}

	Timer.timestamp(passEncoder,"color-end");

	passEncoder.endPass();
	Timer.resolve(renderer, commandEncoder);
	device.queue.submit([commandEncoder.finish()]);

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
		
		device.queue.writeBuffer(
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
	renderer.device.queue.submit([commandBuffer]);

}

function prepare(renderer, octree, camera){
	let {device} = renderer;

	let {pipeline_prepare, uniformBuffer, ssbo_prep_meta, ssbo_prep, ssbo_point_id} = getProgressiveState(renderer);
	let {ssbo_colors} = getColorState(renderer);

	const commandEncoder = device.createCommandEncoder();
	let passEncoder = commandEncoder.beginComputePass();

	Timer.timestamp(passEncoder,"prepare-start");

	passEncoder.setPipeline(pipeline_prepare);

	{

		let vbo_position = gpuAttributeBuffers.get("position").vbo;

		let bindGroup = device.createBindGroup({
			layout: pipeline_prepare.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: {buffer: ssbo_prep_meta}},
				{binding: 2, resource: {buffer: ssbo_colors}},
				{binding: 3, resource: {buffer: ssbo_prep}},
				// {binding: 4, resource: {buffer: ssbo_prep_col}},
				{binding: 5, resource: {buffer: ssbo_point_id}},
				{binding: 6, resource: {buffer: vbo_position}},
			]
		});

		passEncoder.setBindGroup(0, bindGroup);

		let size = renderer.getSize();
		let groups = Math.floor(size.width * size.height) / 128;
		passEncoder.dispatch(groups, 1, 1);
	}

	Timer.timestamp(passEncoder,"prepare-end");

	passEncoder.endPass();
	Timer.resolve(renderer, commandEncoder);
	device.queue.submit([commandEncoder.finish()]);

	// renderer.readBuffer(ssbo_prep_meta, 0, 4).then(buffer => {
	// 	let numPoints = new Uint32Array(buffer)[0];

	// 	console.log(`numPoints: ${numPoints}`);
	// });

}

function reproject(renderer, octree, camera){
	let {device} = renderer;
	let {pipeline_reproject, ssbo_prep_meta, ssbo_prep,ssbo_point_id} = getProgressiveState(renderer);
	let ssbo_depth = getDepthState(renderer).ssbo;
	let {ssbo_colors, uniformBuffer} = getColorState(renderer);

	const commandEncoder = device.createCommandEncoder();
	let passEncoder = commandEncoder.beginComputePass();

	Timer.timestamp(passEncoder,"reproject-start");

	passEncoder.setPipeline(pipeline_reproject);

	{
		let bindGroup = device.createBindGroup({
			layout: pipeline_reproject.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: {buffer: ssbo_depth}},
				{binding: 2, resource: {buffer: ssbo_colors}},
				{binding: 3, resource: {buffer: ssbo_point_id}},
				{binding: 4, resource: {buffer: ssbo_prep}},
			]
		});

		passEncoder.setBindGroup(0, bindGroup);

		let buffer = ssbo_prep_meta;
		let offset = 4;
		passEncoder.dispatchIndirect(buffer, offset);
	}

	Timer.timestamp(passEncoder,"reproject-end");

	passEncoder.endPass();
	Timer.resolve(renderer, commandEncoder);
	device.queue.submit([commandEncoder.finish()])

}

let firstPoint = 0;
let fillBudget = 10_000_000;
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

// {
// 	let el = document.body;

// 	let elButton = document.createElement("input");
// 	elButton.value = "step";
// 	elButton.type = "button";
// 	elButton.style.zIndex = 10000;
// 	elButton.style.position = "absolute";
// 	elButton.style.left = "10px";
// 	elButton.style.top = "300px";
// 	elButton.onclick = () => {
// 		makeStep = true;
// 	};

// 	el.appendChild(elButton);
// }

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

	step(octree);


	resetBuffers(renderer);
	updateUniforms(renderer, octree, camera);

	for(let node of octree.visibleNodes){
		getGpuBuffers(renderer, node.geometry);
	}

	depthPass(renderer, octree, camera);
	colorPass(renderer, octree, camera);
	reproject(renderer, octree, camera);
	resolve(renderer, octree, camera);
	prepare(renderer, octree, camera);

	frame++;

	return target.colorAttachments[0].texture;
}