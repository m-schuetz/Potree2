
// after https://github.com/austinEng/webgpu-samples/blob/master/src/examples/rotatingCube.ts

import { mat4, vec3 } from '../../libs/gl-matrix.js';
import * as shaders from "../prototyping/shaders.js";
import {renderBoundingBoxes} from "../modules/drawCommands/renderBoundingBoxes.js";


class Draws{

	constructor(){
		this.boxes = [];
		this.spheres = [];
	}

	reset(){
		this.boxes = [];
		this.spheres = [];
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
	}

	async init(){
		this.adapter = await navigator.gpu.requestAdapter();
		this.device = await this.adapter.requestDevice();

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

	drawBoundingBox(position, size, color){
		this.draws.boxes.push([position, size, color]);
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

		let commandEncoderDescriptor = {
			measureExecutionTime: true,
		};

		const commandEncoder = this.device.createCommandEncoder(commandEncoderDescriptor);
		const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

		return {commandEncoder, passEncoder, renderPassDescriptor};
	}

	renderDrawCommands(pass, camera){
		renderBoundingBoxes(this, pass, this.draws.boxes, camera);
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