
// after https://github.com/austinEng/webgpu-samples/blob/master/src/examples/rotatingCube.ts

import { mat4, vec3 } from '../../libs/gl-matrix.js';
import * as shaders from "../prototyping/shaders.js";

export class Renderer{

	constructor(){
		this.adapter = null;
		this.device = null;
		this.canvas = null;
		this.context = null;
		this.swapChain = null;
		this.swapChainFormat = null;
		this.depthTexture = null;
		this.state = new Map();
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

	createBuffer(data){

		let vbos = [];

		for(let entry of data.buffers){
			let {name, buffer} = entry;

			let vbo = this.device.createBuffer({
				size: buffer.byteLength,
				usage: GPUBufferUsage.VERTEX,
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

		return vbos;
	}

	createPipeline(vbos){

		const pipeline = this.device.createRenderPipeline({
			vertexStage: {
				module: this.device.createShaderModule({code: shaders.vs}),
				entryPoint: "main",
			},
			fragmentStage: {
				module: this.device.createShaderModule({code: shaders.fs}),
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
					{ // position
						arrayStride: 3 * 4,
						attributes: [{ 
							shaderLocation: 0,
							offset: 0,
							format: "float3",
						}],
					},{ // color
						arrayStride: 4 * 4,
						attributes: [{ 
							shaderLocation: 1,
							offset: 0,
							format: "float4",
						}],
					},
				],
			},
			rasterizationState: {
				cullMode: "none",
			},
			colorStates: [{
					format: "bgra8unorm",
			}],
		});

		return pipeline;
	}

	getState(node){

		let state = this.state.get(node);

		if(!state){
			let vbos = this.createBuffer(node);
			let pipeline = this.createPipeline();

			const uniformBufferSize = 4 * 16; 

			const uniformBuffer = this.device.createBuffer({
				size: uniformBufferSize,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			});

			const uniformBindGroup = this.device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: [{
					binding: 0,
					resource: {buffer: uniformBuffer},
				}],
			});

			state = {
				vbos: vbos,
				pipeline: pipeline,
				uniformBuffer: uniformBuffer,
				uniformBindGroup: uniformBindGroup,
			};

			this.state.set(node, state);

		}

		return state;
	}

	start(){

		let scale = window.devicePixelRatio
		this.setSize(scale * this.canvas.clientWidth, scale * this.canvas.clientHeight);

		let renderPassDescriptor = {
			colorAttachments: [
				{
					attachment: this.swapChain.getCurrentTexture().createView(),
					loadValue: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
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

		return {commandEncoder, renderPassDescriptor};
	}

	finish(pass){
		this.device.defaultQueue.submit([pass.commandEncoder.finish()]);
	}

	render(pass, node, camera){

		let {device, swapChain, depthTexture} = this;
		let size = this.getSize();

		let state = this.getState(node);

		let {uniformBuffer, pipeline, uniformBindGroup} = state;

		camera.aspect = size.width / size.height;
		camera.updateProj();

		let world = mat4.create();

		mat4.translate(world, world, vec3.fromValues(-0.5, -0.5, -0.5));
		// mat4.rotate(
		// 	world,
		// 	world,
		// 	Date.now() / 1000,
		// 	vec3.fromValues(0, 1, 0)
		// );

		let view = camera.view;
		let proj = camera.proj;

		let transformationMatrix = mat4.create();
		mat4.multiply(transformationMatrix, view, world);
		mat4.multiply(transformationMatrix, proj, transformationMatrix);

		//const transformationMatrix = getTransformationMatrix();

		device.defaultQueue.writeBuffer(
			uniformBuffer,
			0,
			transformationMatrix.buffer,
			transformationMatrix.byteOffset,
			transformationMatrix.byteLength
		);

		{
			let passEncoder = pass.commandEncoder.beginRenderPass(pass.renderPassDescriptor);
			passEncoder.setPipeline(pipeline);
			passEncoder.setBindGroup(0, uniformBindGroup);

			let vbos = state.vbos;
			for(let i = 0; i < vbos.length; i++){
				passEncoder.setVertexBuffer(i, vbos[i].vbo);
			}

			passEncoder.draw(node.vertexCount, 1, 0, 0);
			passEncoder.endPass();
		}

		// device.defaultQueue.submit([commandEncoder.finish()]);

	}


};