
// after https://github.com/austinEng/webgpu-samples/blob/master/src/examples/rotatingCube.ts

import * as shaders from "../prototyping/shaders.js";
import {renderBoundingBoxes} from "../modules/drawCommands/renderBoundingBoxes.js";
import {renderLines} from "../modules/drawCommands/renderLines.js";


class Draws{

	constructor(){
		this.boxes = [];
		this.spheres = [];
		this.lines = [];
	}

	reset(){
		this.boxes = [];
		this.spheres = [];
		this.lines = [];
	}

};

export class Renderer{

	constructor(){
		this.adapter = null;
		this.device = null;
		this.canvas = null;
		this.context = null;
		this.swapChain = null;
		this.swapChainFormat = null;
		this.depthTexture = null;
		this.draws = new Draws();

		this.buffers = new Map();
	}

	async init(){
		this.adapter = await navigator.gpu.requestAdapter();
		this.device = await this.adapter.requestDevice({
			extensions: ["timestamp-query"],
		});
		this.canvas = document.getElementById("canvas");
		this.context = this.canvas.getContext("gpupresent");

		this.swapChainFormat = "bgra8unorm";
		this.swapChain = this.context.configureSwapChain({
			device: this.device,
			format: this.swapChainFormat,
			usage: GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.COPY_DST,
		});

		this.depthTexture = this.device.createTexture({
			size: {
				width: this.canvas.width,
				height: this.canvas.height,
				depth: 1,
			},
			format: "depth24plus-stencil8",
			usage: GPUTextureUsage.OUTPUT_ATTACHMENT,
		});
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

			this.depthTexture = this.device.createTexture({
				size: {
					width: this.canvas.width,
					height: this.canvas.height,
					depth: 1,
				},
				format: "depth24plus-stencil8",
				usage: GPUTextureUsage.OUTPUT_ATTACHMENT,
			});
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
		this.device.defaultQueue.submit([copyCommands]);

		await target.mapAsync(GPUMapMode.READ);

		const copyArrayBuffer = target.getMappedRange();

		let cloned = copyArrayBuffer.slice();

		target.unmap();

		return cloned;
		
	}

	createTexture(width, height, params = {}){

		let format = params.format ?? "rgba8uint";

		let texture = this.device.createTexture({
			size: [width, height, 1],
			format: format,
			usage: 
				GPUTextureUsage.STORAGE
				| GPUTextureUsage.SAMPLED 
				| GPUTextureUsage.COPY_SRC 
				| GPUTextureUsage.COPY_DST 
				| GPUTextureUsage.OUTPUT_ATTACHMENT
				// | GPUTextureUsage.STORAGE
				// | GPUTextureUsage.COPY_SRC 
				// | GPUTextureUsage.COPY_DST 
				// | GPUTextureUsage.OUTPUT_ATTACHMENT,
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

	getGpuBuffers(geometry){

		let buffers = this.buffers.get(geometry);

		if(!buffers){

			let {device} = renderer;

			let vbos = [];

			for(let entry of geometry.buffers){
				let {name, buffer} = entry;

				let vbo = device.createBuffer({
					size: buffer.byteLength,
					usage: GPUBufferUsage.VERTEX | GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE,
					mappedAtCreation: true,
				});

				let type = buffer.constructor;
				new type(vbo.getMappedRange()).set(buffer);
				vbo.unmap();

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

	drawBoundingBox(position, size, color){
		this.draws.boxes.push([position, size, color]);
	}

	drawLine(start, end, color){
		this.draws.lines.push([start, end, color]);
	}

	start(){

		let scale = window.devicePixelRatio
		this.setSize(scale * this.canvas.clientWidth, scale * this.canvas.clientHeight);

		let renderPassDescriptor = {
			colorAttachments: [
				{
					attachment: this.swapChain.getCurrentTexture().createView(),
					loadValue: { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
				},
			],
			depthStencilAttachment: {
				attachment: this.depthTexture.createView(),

				depthLoadValue: 1.0,
				depthStoreOp: "store",
				stencilLoadValue: 0,
				stencilStoreOp: "store",
			},
			sampleCount: 1,
		};

		const commandEncoder = this.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

		return {commandEncoder, passEncoder, renderPassDescriptor};
	}

	renderDrawCommands(pass, camera){
		renderBoundingBoxes(this, pass, this.draws.boxes, camera);
		renderLines(this, pass, this.draws.lines, camera);
	}

	finish(pass){

		pass.passEncoder.endPass();

		let commandBuffer = pass.commandEncoder.finish();
		this.device.defaultQueue.submit([commandBuffer]);

		// not yet available?
		// https://github.com/gpuweb/gpuweb/issues/1325
		// commandBuffer.executionTime.then( (e) => {
		// 	console.log(e);
		// });

		this.draws.reset();

	}

};