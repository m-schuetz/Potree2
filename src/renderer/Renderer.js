
// after https://github.com/austinEng/webgpu-samples/blob/master/src/examples/rotatingCube.ts

import * as shaders from "../prototyping/shaders.js";
import {render as renderBoxes} from "../modules/drawCommands/renderBoxes.js";
import {render as renderBoundingBoxes} from "../modules/drawCommands/renderBoundingBoxes.js";
import {render as renderPoints} from "../modules/drawCommands/renderPoints.js";
import {render as renderQuads} from "../modules/drawCommands/renderQuads.js";
import {render as renderLines} from "../modules/drawCommands/renderLines.js";
import {render as renderMeshes} from "../modules/drawCommands/renderMeshes.js";
import * as Timer from "./Timer.js";
import {writeBuffer} from "./writeBuffer.js";
import {fillBuffer} from "./fillBuffer.js";
import {readPixels, readDepth} from "../renderer/readPixels.js";
import {RenderTarget} from "potree";


class Draws{

	constructor(){
		this.boundingBoxes = [];
		this.boxes = [];
		this.spheres = [];
		this.lines = [];
		this.points = [];
		this.quads = [];
		this.meshes = [];
	}

	reset(){
		this.boundingBoxes = [];
		this.boxes = [];
		this.spheres = [];
		this.lines = [];
		this.points = [];
		this.quads = [];
		this.meshes = [];
	}

};

export class Renderer{

	constructor(){
		this.adapter = null;
		this.device = null;
		this.canvas = null;
		this.context = null;
		this.swapChainFormat = null;
		this.draws = new Draws();
		this.currentBindGroup = -1;
		this.frameCounter = 0;

		this.swapChain = null;
		this.depthTexture = null;
		this.screenbuffer = null;

		this.framebuffers = new Map();
		this.buffers = new Map();
		this.typedBuffers = new Map();
		this.textures = new Map();
	}

	async init(){
		this.adapter = await navigator.gpu.requestAdapter();
		this.device = await this.adapter.requestDevice({
			requiredFeatures: ["timestamp-query"],
		});
		this.canvas = document.getElementById("canvas");
		this.context = this.canvas.getContext("webgpu");

		this.swapChainFormat = "bgra8unorm";
		this.context.configure({
			device: this.device,
			format: this.swapChainFormat,
			usage: GPUTextureUsage.RENDER_ATTACHMENT 
				| GPUTextureUsage.COPY_DST 
				| GPUTextureUsage.COPY_SRC
				| GPUTextureUsage.TEXTURE_BINDING,
		});

		this.swapChain = this.context.getCurrentTexture();

		let size = this.getSize();
		this.depthTexture = this.device.createTexture({
			size: {width: size.width, height: size.height},
			format: "depth32float",
			usage: GPUTextureUsage.TEXTURE_BINDING 
				| GPUTextureUsage.COPY_SRC 
				| GPUTextureUsage.COPY_DST 
				| GPUTextureUsage.RENDER_ATTACHMENT,
		});

		// this.screenbuffer = RenderTarget.create();

		{
			this.screenbuffer = Object.create(RenderTarget.prototype);
		}

		this.updateScreenbuffer();
	}

	updateScreenbuffer(){
		let size = this.getSize();

		this.screenbuffer.colorAttachments = [{
			descriptor: {
				size: [size.width, size.height],
				format: this.swapChainFormat,
				usage: GPUTextureUsage.TEXTURE_BINDING 
					| GPUTextureUsage.COPY_SRC 
					| GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.RENDER_ATTACHMENT,
			},
			texture: this.context.getCurrentTexture(),
		}];
		this.screenbuffer.depth = {
			descriptor: {
				size: [size.width, size.height],
				format: "depth32float",
				usage: GPUTextureUsage.TEXTURE_BINDING 
					| GPUTextureUsage.COPY_SRC 
					| GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.RENDER_ATTACHMENT,
			},
			texture: this.depthTexture,
		};
		this.screenbuffer.size = [size.width, size.height];
	}

	getSize(){
		return {
			width: this.canvas.width,
			height: this.canvas.height
		};
	}

	setSize(width, height){

		width = Math.min(width, 7680);
		height = Math.min(height, 4320);

		let resized = this.canvas.width !== width || this.canvas.height !== height;

		if(resized){
			this.canvas.width = width;
			this.canvas.height = height;

			let size = {width, height};

			this.context.configure({
				device: this.device,
				format: this.swapChainFormat,
				usage: GPUTextureUsage.RENDER_ATTACHMENT 
					| GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.COPY_SRC
					| GPUTextureUsage.TEXTURE_BINDING,
			});

			this.depthTexture = this.device.createTexture({
				size: size,
				format: "depth32float",
				usage: GPUTextureUsage.TEXTURE_BINDING 
					| GPUTextureUsage.COPY_SRC 
					| GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.RENDER_ATTACHMENT,
			});

			this.updateScreenbuffer();
		}
	}

	async readBuffer(source, start, size){
		const target = this.device.createBuffer({
			size: size,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});

		let sourceOffset = start;
		let targetOffset = 0;
		let targetSize = size;

		const copyEncoder = this.device.createCommandEncoder();
		copyEncoder.copyBufferToBuffer(
			source, sourceOffset,
			target, targetOffset, targetSize);

		// Submit copy commands.
		const copyCommands = copyEncoder.finish();
		this.device.queue.submit([copyCommands]);

		await target.mapAsync(GPUMapMode.READ);

		const copyArrayBuffer = target.getMappedRange();

		let cloned = copyArrayBuffer.slice();

		target.unmap();

		return cloned;
	}

	async readPixels(texture, x, y, width, height){

		let bytesPerRow = width * 4; 
		
		// "bytesPerRow must be a multiple of 256"
		bytesPerRow = Math.ceil(bytesPerRow / 256) * 256;

		let size = bytesPerRow * height;

		// copyTextureToBuffer
		const buffer = this.device.createBuffer({
			size: size,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});

		let source = {
			texture: texture,
			origin: {x, y, z: 0},
		};

		let destination = {
			buffer,
			bytesPerRow: bytesPerRow,
		};

		let copySize = {width, height, depthOrArrayLayers: 1};

		const copyEncoder = this.device.createCommandEncoder();
		copyEncoder.copyTextureToBuffer(source, destination, copySize);

		// Submit copy commands.
		const copyCommands = copyEncoder.finish();
		this.device.queue.submit([copyCommands]);

		await buffer.mapAsync(GPUMapMode.READ);

		const copyArrayBuffer = buffer.getMappedRange();

		let cloned = copyArrayBuffer.slice();

		buffer.unmap();

		return cloned;
	}

	createComputePipeline(args){

		let {device} = this;
		let code = args.code;
		let entryPoint = args.entryPoint ?? "main";

		let pipeline =  device.createComputePipeline({
			compute: {
				module: device.createShaderModule({code: code}),
				entryPoint: entryPoint,
			},
		});

		return pipeline;
	}

	runCompute({code, bindGroups, dispatchGroups, entryPoint}){

		let {device} = this;
		let pipeline = this.createComputePipeline({code, entryPoint});

		

		const commandEncoder = device.createCommandEncoder();
		const passEncoder = commandEncoder.beginComputePass();

		passEncoder.setPipeline(pipeline);

		for(let bindGroupItem of bindGroups){

			let bindGroup = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(bindGroupItem.location),
				entries: bindGroupItem.entries,
			});

			passEncoder.setBindGroup(bindGroupItem.location, bindGroup);
		}

		passEncoder.dispatch(...dispatchGroups);
		passEncoder.endPass();
		
		device.queue.submit([commandEncoder.finish()]);
	}

	createTextureFromArray(array, width, height){
		let texture = this.createTexture(width, height, {format: "rgba8unorm"});

		let raw = new Uint8ClampedArray(array);
		let imageData = new ImageData(raw, width, height);

		createImageBitmap(imageData).then(bitmap => {

			this.device.queue.copyExternalImageToTexture(
				{source: bitmap}, 
				{texture: texture},
				[bitmap.width, bitmap.height, 1]
			);

			// this.device.queue.copyImageBitmapToTexture(
			// 	{imageBitmap: bitmap}, {texture: texture},
			// 	[bitmap.width, bitmap.height, 1]
			// );

		});

		return texture;
	}

	createTexture(width, height, params = {}){

		let format = params.format ?? "rgba8uint";

		let texture = this.device.createTexture({
			size: [width, height, 1],
			format: format,
			usage: 
				GPUTextureUsage.STORAGE
				| GPUTextureUsage.TEXTURE_BINDING 
				| GPUTextureUsage.COPY_SRC 
				| GPUTextureUsage.COPY_DST 
				| GPUTextureUsage.RENDER_ATTACHMENT
		});

		return texture;
	}

	createBuffer(size){

		let buffer = this.device.createBuffer({
			size: size,
			usage: GPUBufferUsage.VERTEX 
				| GPUBufferUsage.STORAGE
				| GPUBufferUsage.COPY_SRC
				| GPUBufferUsage.COPY_DST
				| GPUBufferUsage.UNIFORM,
		});

		return buffer;
	}

	writeBuffer(args){
		writeBuffer(this, args);
	}

	fillBuffer(buffer, value, numU32Elements){
		fillBuffer(this, buffer, value, numU32Elements);
	}
	
	getGpuTexture(image){

		let gpuTexture = this.textures.get(image);

		if(!gpuTexture){
			let {device} = this;

			let width = image?.width ?? 128;
			let height = image?.height ?? 128;

			gpuTexture = device.createTexture({
				size: [width, height, 1],
				format: "rgba8unorm",
				usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
			});

			if(image){
				device.queue.copyImageBitmapToTexture(
					{imageBitmap: image}, {texture: gpuTexture},
					[image.width, image.height, 1]
				);
			}

			this.textures.set(image, gpuTexture);
		}

		return gpuTexture;
	}

	getGpuBuffer(typedArray){
		let buffer = this.typedBuffers.get(typedArray);
		
		if(!buffer){
			let {device} = renderer;
			
			let vbo = device.createBuffer({
				size: typedArray.byteLength,
				usage: GPUBufferUsage.VERTEX 
					| GPUBufferUsage.INDEX  
					| GPUBufferUsage.COPY_DST 
					| GPUBufferUsage.STORAGE,
				mappedAtCreation: true,
			});

			let type = typedArray.constructor;
			new type(vbo.getMappedRange()).set(typedArray);
			vbo.unmap();

			buffer = vbo;

			this.typedBuffers.set(typedArray, buffer);
		}

		return buffer;
	}

	getGpuBuffers(geometry){

		let buffers = this.buffers.get(geometry);

		if(!buffers){

			let {device} = renderer;

			let vbos = [];

			for(let entry of geometry.buffers){
				let {name, buffer} = entry;
				
				let vbo = this.getGpuBuffer(buffer);

				vbos.push({
					name: name,
					vbo: vbo,
				});
			}

			this.buffers.set(geometry, vbos);

			buffers = vbos;
		}

		return buffers;
	}

	getFramebuffer(id){

		if(this.framebuffers.has(id)){
			return this.framebuffers.get(id);
		}else{

			let size = [128, 128, 1];
			let descriptor = {
				size: size,
				colorDescriptors: [{
					size: size,
					format: this.swapChainFormat,
					usage: GPUTextureUsage.TEXTURE_BINDING 
						// | GPUTextureUsage.COPY_SRC 
						// | GPUTextureUsage.COPY_DST 
						| GPUTextureUsage.RENDER_ATTACHMENT,
				}],
				depthDescriptor: {
					size: size,
					format: "depth32float",
					usage: GPUTextureUsage.TEXTURE_BINDING 
						// | GPUTextureUsage.COPY_SRC 
						// | GPUTextureUsage.COPY_DST 
						| GPUTextureUsage.RENDER_ATTACHMENT,
				}
			};

			let framebuffer = new RenderTarget(this, descriptor);
			
			this.framebuffers.set(id, framebuffer);

			return framebuffer;
		}

	}

	drawBoundingBox(position, size, color){
		this.draws.boundingBoxes.push([position, size, color]);
	}

	drawBox(position, size, color){
		this.draws.boxes.push([position, size, color]);
	}

	drawMesh(args){
		this.draws.meshes.push(args);
	}

	drawLine(start, end, color){

		if(start instanceof Array){
			start = new Vector3(...start);
		}

		if(end instanceof Array){
			end = new Vector3(...end);
		}

		if(color instanceof Array){
			color = new Vector3(...color);
		}

		this.draws.lines.push([start, end, color]);
	}

	drawPoints(positions, colors){
		this.draws.points.push({positions, colors});
	}

	drawQuads(positions, colors){
		this.draws.quads.push({positions, colors});
	}

	start(){

		// let scale = window.devicePixelRatio;
		// this.setSize(scale * this.canvas.clientWidth, scale * this.canvas.clientHeight);
		this.setSize(this.canvas.clientWidth, this.canvas.clientHeight);

		this.updateScreenbuffer();

		// let renderPassDescriptor = {
		// 	colorAttachments: [
		// 		{
		// 			view: this.swapChain.getCurrentTexture().createView(),
		// 			loadValue: { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
		// 		},
		// 	],
		// 	depthStencilAttachment: {
		// 		view: this.depthTexture.createView(),

		// 		depthLoadValue: 0.0,
		// 		depthStoreOp: "store",
		// 		stencilLoadValue: 0,
		// 		stencilStoreOp: "store",
		// 	},
		// 	sampleCount: 1,
		// };

		// const commandEncoder = this.device.createCommandEncoder();
		// const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

		// return {commandEncoder, passEncoder, renderPassDescriptor};
	}

	finish(){

		// pass.passEncoder.endPass();

		// Timer.resolve(renderer, pass.commandEncoder);

		// let commandBuffer = pass.commandEncoder.finish();
		// this.device.queue.submit([commandBuffer]);



		// not yet available?
		// https://github.com/gpuweb/gpuweb/issues/1325
		// commandBuffer.executionTime.then( (e) => {
		// 	console.log(e);
		// });

		this.draws.reset();
		this.currentBindGroup = -1;
		this.frameCounter++;
	}

	getNextBindGroup(){
		this.currentBindGroup++;

		return this.currentBindGroup;
	}

	renderDrawCommands(drawstate){
		renderBoxes(this.draws.boxes, drawstate);
		renderBoundingBoxes(this.draws.boundingBoxes, drawstate);
		renderPoints(this.draws.points, drawstate);
		renderQuads(this.draws.quads, drawstate);
		renderMeshes(this.draws.meshes, drawstate);
		renderLines(this.draws.lines, drawstate);
	}

	update(){

	}

	// render(scene, camera){
		
	// }

};