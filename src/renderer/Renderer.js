
// after https://github.com/austinEng/webgpu-samples/blob/master/src/examples/rotatingCube.ts

import * as shaders from "../prototyping/shaders.js";
import {render as renderBoxes} from "../modules/drawCommands/renderBoxes.js";
import {render as renderSpheres} from "../modules/drawCommands/renderSpheres.js";
import {render as renderBoundingSpheres} from "../modules/drawCommands/renderBoundingSpheres.js";
import {render as renderBoundingBoxes} from "../modules/drawCommands/renderBoundingBoxes.js";
import {render as renderPoints} from "../modules/drawCommands/renderPoints.js";
import {render as renderQuads} from "../modules/drawCommands/renderQuads.js";
import {render as renderLines} from "../modules/drawCommands/renderLines.js";
import {render as renderMeshes} from "../modules/drawCommands/renderMeshes.js";
import {render as renderVoxels} from "../modules/drawCommands/renderVoxels.js";
import * as Timer from "./Timer.js";
import {writeBuffer} from "./writeBuffer.js";
import {fillBuffer} from "./fillBuffer.js";
import {readPixels, readDepth} from "../renderer/readPixels.js";
import {RenderTarget} from "potree";
import {ChunkedBuffer} from "potree";

let dbgSet = new Set();


let dbg_numNewlyCreatedBuffers = 0;
let dbg_newBufferBytes = 0;

class Draws{

	constructor(){
		this.boundingBoxes = [];
		this.boxes = [];
		this.spheres = [];
		this.lines = [];
		this.points = [];
		this.quads = [];
		this.voxels = [];
		this.meshes = [];
	}

	reset(){
		this.boundingBoxes = [];
		this.boxes = [];
		this.spheres = [];
		this.lines = [];
		this.points = [];
		this.quads = [];
		this.voxels = [];
		this.meshes = [];
	}

};

export class TimestampEntry{
	constructor(){
		this.startIndex = 0;
		this.endIndex = 0;
		this.label = "";
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
		this.drawListeners = [];
		this.currentBindGroup = -1;
		this.frameCounter = 0;

		this.defaultSampler = null;
		this.defaultTexture = null;

		this.timestamps = {
			enabled:               false,
			querySet:              null,
			resolveBuffer:         null,
			resultBuffer:          null,
			resultBufferPool:      [],
			numResolveRequested:   0,
			entries:               []
		};

		this.depthTexture  = null;
		this.screenbuffer  = null;

		this.framebuffers  = new Map();
		this.buffers       = new Set();
		this.cpuGpuBuffers = new Map();
		this.textures      = new Set();
	}

	createResourceReport(){

		let framebufferBytes  = 0;
		let bufferBytes      = -1;
		let cpuGpuBufferBytes  = 0;
		let textureBytes      = 0;

		for(let [cpuBuffer, gpuBuffer] of this.cpuGpuBuffers){
			cpuGpuBufferBytes += cpuBuffer.byteLength;
		}

		for(let buffer of this.buffers){
			bufferBytes += buffer.size;
		}

		for(let texture of this.textures){
			// TODO: assume every pixel needs 4 byte, for now
			textureBytes += (texture.width * texture.height) * 4;
		}

		let msg =  `type              count           bytes\n`;
		msg +=     `=======================================\n`;

		{
			let count = this.cpuGpuBuffers.size.toLocaleString().padStart(7);
			let strBytes = `${(cpuGpuBufferBytes / 1_000_000).toFixed(1)} MB`.padStart(12);
			msg += `CPU-GPU Buffers   ${count}    ${strBytes}\n`;
		}

		{
			let count = this.buffers.size.toLocaleString().padStart(7);
			let strBytes = `${(bufferBytes / 1_000_000).toFixed(1)} MB`.padStart(12);
			msg += `buffers           ${count}    ${strBytes}\n`;
		}

		{
			let count = this.textures.size.toLocaleString().padStart(7);
			let strBytes = `${(textureBytes / 1_000_000).toFixed(1)} MB`.padStart(12);
			msg += `textures          ${count}    ${strBytes}\n`;
		}


		return msg;
	}

	async init(){
		this.adapter = await navigator.gpu.requestAdapter({"powerPreference": "high-performance"});

		this.timestamps.enabled = this.adapter.features.has("timestamp-query");
		// this.timestamps.enabled = false;

		let requiredFeatures = [];

		if(this.timestamps.enabled){
			requiredFeatures.push("timestamp-query");
		}

		this.device = await this.adapter.requestDevice({
			requiredFeatures: requiredFeatures,
			requiredLimits: {
				maxStorageBufferBindingSize: 1_073_741_824,
				maxBufferSize: 1_073_741_824,
				// maxBindGroups: 16,
			}
		});
		
		Timer.setEnabled(false);
		// Timer.setEnabled(this.timestampQueriesEnabled);

		if(this.timestamps.enabled){

			let MAX_ENTRIES = 256;

			this.timestamps.querySet = this.device.createQuerySet({
				type: 'timestamp',
				count: MAX_ENTRIES,
			});

			// Create some resolve buffers
			// Metal only allows 256 byte offsets, so we need to allocate 256 instead of 2*8 byte per timestamp entry
			// for(let i = 0; i < 2; i++){

			//  	let buffer = this.device.createBuffer({
			// 		size: 256 * MAX_ENTRIES,
			// 		usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
			// 	});

			// 	this.timestamps.resolveBufferPool.push(buffer);
			// }
			this.timestamps.resolveBuffer = this.device.createBuffer({
				size: 256 * MAX_ENTRIES,
				usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
			});

			// Metal only allows 256 byte offsets, so we need to allocate 256 instead of 2*8 byte per timestamp entry
			// this.timestamps.resultBuffer = this.device.createBuffer({
			// 	size: 256 * MAX_ENTRIES,
			// 	usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			// });
			for(let i = 0; i < 2; i++){

			 	let buffer = this.device.createBuffer({
					size: 256 * MAX_ENTRIES,
					usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
				});

				this.timestamps.resultBufferPool.push(buffer);
			}
		}

		this.canvas = document.getElementById("canvas");
		this.context = this.canvas.getContext("webgpu");

		this.canvas.setAttribute("tabindex", 0);

		this.swapChainFormat = "bgra8unorm";
		this.context.configure({
			device: this.device,
			format: this.swapChainFormat,
			usage: GPUTextureUsage.RENDER_ATTACHMENT 
				| GPUTextureUsage.COPY_DST 
				| GPUTextureUsage.COPY_SRC
				| GPUTextureUsage.TEXTURE_BINDING,
			alphaMode: "opaque",
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

		this.screenbuffer = Object.create(RenderTarget.prototype);
		this.defaultTexture = this.createTextureFromArray(new Uint8Array([255, 0, 0, 255]), 1, 1);

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

		let clamp = (value, min, max) => Math.max(Math.min(value, max), min);

		width = clamp(width, 128, 7680);
		height = clamp(height, 128, 4320);

		let resized = this.canvas.width !== width || this.canvas.height !== height;

		if(resized)
		{
			this.canvas.width = width;
			this.canvas.height = height;

			let size = {width, height};

			// console.log(`configure`, {width, height});
			// this.context.configure({
			// 	device: this.device,
			// 	format: this.swapChainFormat,
			// 	usage: GPUTextureUsage.RENDER_ATTACHMENT 
			// 		| GPUTextureUsage.COPY_DST 
			// 		| GPUTextureUsage.COPY_SRC
			// 		| GPUTextureUsage.TEXTURE_BINDING,
			// 	size: {width, height},
			// 	alphaMode: "opaque",
			// });

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

		x = Math.max(x, 0);
		y = Math.max(y, 0);

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

		let source_u8 = new Uint8Array(cloned);
		let target_u8 = new Uint8Array(4 * width * height);

		for(let px = 0; px < width; px++){
		for(let py = 0; py < height; py++){

			let source_offset = 4 * px + bytesPerRow * py;
			let target_offset = 4 * px + 4 * width * py;

			target_u8[target_offset + 0] = source_u8[source_offset + 0];
			target_u8[target_offset + 1] = source_u8[source_offset + 1];
			target_u8[target_offset + 2] = source_u8[source_offset + 2];
			target_u8[target_offset + 3] = source_u8[source_offset + 3];
		}
		}

		return target_u8.buffer;
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
		passEncoder.end();
		
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
		});

		this.textures.add(texture);

		return texture;
	}

	createTexture(width, height, params = {}){

		let format = params.format ?? "rgba8uint";
		let label  = params.label ?? "missing texture label";

		let texture = this.device.createTexture({
			size: [width, height, 1],
			format: format,
			arrayLayerCount: 1,
			mipLevelCount: 1,
			sampleCount: 1,
			dimension: "2d",
			label: label,
			usage: 
				GPUTextureUsage.STORAGE_BINDING
				| GPUTextureUsage.TEXTURE_BINDING 
				| GPUTextureUsage.COPY_SRC 
				| GPUTextureUsage.COPY_DST 
				| GPUTextureUsage.RENDER_ATTACHMENT
		});

		this.textures.add(texture);

		return texture;
	}

	createBuffer({size, usage}){

		if(!usage){
			usage = GPUBufferUsage.VERTEX 
				| GPUBufferUsage.STORAGE
				| GPUBufferUsage.COPY_SRC
				| GPUBufferUsage.COPY_DST
				| GPUBufferUsage.UNIFORM
		}

		console.log(`createBuffer(${size.toLocaleString()})`);

		let buffer = this.device.createBuffer({size, usage});

		this.buffers.add(buffer);

		return buffer;
	}

	createChunkedBuffer(size, chunkSize){
		debugger;
		let buffer = new ChunkedBuffer(size, chunkSize, this);

		return buffer;
	}

	dispose(resource){
		if(resource instanceof GPUTexture){
			
			if(this.textures.has(resource)){
				this.textures.delete(resource);
				resource.destroy();
			}else{
				console.error("the tracker did not know of this GPUTexture");
			}

		}
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
				usage: GPUTextureUsage.TEXTURE_BINDING 
					| GPUTextureUsage.COPY_DST
					| GPUTextureUsage.RENDER_ATTACHMENT
					| GPUTextureUsage.COPY_DST,
			});

			if(image){
				device.queue.copyExternalImageToTexture(
					{source: image},
					{texture: gpuTexture},
					[image.width, image.height, 1]
					// [128, 128, 1]
				);
			}

			this.textures.set(image, gpuTexture);
		}

		return gpuTexture;
	}



	getGpuBuffer(cpuBuffer){
		let gpuBuffer = this.cpuGpuBuffers.get(cpuBuffer);
		
		if(!gpuBuffer){
			let {device} = renderer;

			let byteSize = cpuBuffer.byteLength;

			dbg_numNewlyCreatedBuffers++;
			dbg_newBufferBytes += byteSize;
			
			gpuBuffer = device.createBuffer({
				size: byteSize,
				usage: GPUBufferUsage.VERTEX 
					| GPUBufferUsage.INDEX  
					| GPUBufferUsage.COPY_DST 
					| GPUBufferUsage.COPY_SRC 
					| GPUBufferUsage.STORAGE,
				mappedAtCreation: true,
			});

			if(cpuBuffer instanceof ArrayBuffer){
				new Uint8Array(gpuBuffer.getMappedRange()).set(new Uint8Array(cpuBuffer, 0, byteSize));
			}else{
				new Uint8Array(gpuBuffer.getMappedRange()).set(new Uint8Array(cpuBuffer.buffer, 0, byteSize));
			}

			gpuBuffer.unmap();

			this.cpuGpuBuffers.set(cpuBuffer, gpuBuffer);
		}

		return gpuBuffer;
	}

	disposeGpuBuffer(cpuBuffer){
		let gpuBuffer = this.cpuGpuBuffers.get(cpuBuffer);

		if(gpuBuffer){
			gpuBuffer.destroy();
			this.cpuGpuBuffers.delete(cpuBuffer);
		}
	}

	getGpuBuffers(geometry){

		throw "deprecated";

		debugger;

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

	getDefaultSampler(){
		if(this.defaultSampler){
			return this.defaultSampler;
		}

		this.defaultSampler = this.device.createSampler({
			magFilter: 'nearest',
			minFilter: 'nearest',
			mipmapFilter : 'nearest',
			addressModeU: "repeat",
			addressModeV: "repeat",
			maxAnisotropy: 1,
		});

		return this.defaultSampler;
	}

	getEmptyBuffer(){
		if(this.emptyBuffer){
			return this.emptyBuffer;
		}else{
			let flags = GPUBufferUsage.STORAGE 
				| GPUBufferUsage.COPY_SRC 
				| GPUBufferUsage.COPY_DST 
				| GPUBufferUsage.VERTEX
				| GPUBufferUsage.INDEX;
			this.emptyBuffer = this.device.createBuffer({size: 32, usage: flags});

			return this.emptyBuffer;
		}
	}

	getFramebuffer(id, args = {}){

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
						| GPUTextureUsage.COPY_SRC 
						| GPUTextureUsage.COPY_DST 
						| GPUTextureUsage.RENDER_ATTACHMENT,
					sampleCount: args.sampleCount ?? 1,
				}
				// ,{
				// 	size: size,
				// 	format: this.swapChainFormat,
				// 	usage: GPUTextureUsage.TEXTURE_BINDING 
				// 		| GPUTextureUsage.COPY_SRC 
				// 		| GPUTextureUsage.COPY_DST 
				// 		| GPUTextureUsage.RENDER_ATTACHMENT,
				// 	sampleCount: args.sampleCount ?? 1,
				// }
				],
				depthDescriptor: {
					size: size,
					format: "depth32float",
					usage: GPUTextureUsage.TEXTURE_BINDING 
						| GPUTextureUsage.COPY_SRC 
						| GPUTextureUsage.COPY_DST 
						| GPUTextureUsage.RENDER_ATTACHMENT,
					sampleCount: args.sampleCount ?? 1,
				},
				sampleCount: args.sampleCount ?? 1,
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

	drawSphere(position, radius, args = {}){
		this.draws.spheres.push([position, radius, args]);
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

	drawVoxels(positions, colors, voxelSize){
		this.draws.voxels.push({positions, colors, voxelSize});
	}

	onDraw(callback){
		this.drawListeners.push(callback);
	}

	start(){
		if(this.timestamps.resultBufferPool.length > 0){
			this.timestamps.resultBuffer = this.timestamps.resultBufferPool.pop();
		}

		dbg_numNewlyCreatedBuffers = 0;
		dbg_newBufferBytes = 0;

		let dpr = window.devicePixelRatio;
		this.setSize(this.canvas.clientWidth * dpr, this.canvas.clientHeight * dpr);

		this.updateScreenbuffer();

		this.timestamps.entries = [];
		this.timestamps.numResolveRequested = 0;
	}

	finish(){
		this.draws.reset();
		this.currentBindGroup = -1;
		this.frameCounter++;

		// if(dbg_numNewlyCreatedBuffers > 0){
		// 	let strNewBytes = dbg_newBufferBytes / 1000;
		// 	console.log(`[frame ${this.frameCounter}] newBuffers: ${dbg_numNewlyCreatedBuffers}, newBufferBytes: ${strNewBytes} kb`);
		// }
	}

	getNextBindGroup(){
		this.currentBindGroup++;

		return this.currentBindGroup;
	}

	renderDrawCommands(drawstate){

		renderSpheres(this.draws.spheres, drawstate);
		renderLines(this.draws.lines, drawstate);


		// renderBoxes(this.draws.boxes, drawstate);
		// renderSpheres(this.draws.spheres, drawstate);
		// // renderBoundingSpheres(this.draws.spheres, drawstate);
		// renderBoundingBoxes(this.draws.boundingBoxes, drawstate);
		// // renderPoints(this.draws.points, drawstate);
		// // renderQuads(this.draws.quads, drawstate);
		// // renderVoxels(this.draws.voxels, drawstate);
		// renderMeshes(this.draws.meshes, drawstate);
		// renderLines(this.draws.lines, drawstate);

		for(let listener of this.drawListeners){
			listener(drawstate);
		}
	}

	update(){

	}

	// render(scene, camera){
		
	// }

};