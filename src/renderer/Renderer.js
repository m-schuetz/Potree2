
// after https://github.com/austinEng/webgpu-samples/blob/master/src/examples/rotatingCube.ts

import { mat4, vec3 } from '../../libs/gl-matrix.js';
import * as shaders from "../prototyping/shaders.js";

let cameraDistance = 20;

let aspect = 3 / 4;
let projectionMatrix = mat4.create();
mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, aspect, 0.1, 1000.0);

function getTransformationMatrix() {

	aspect = canvas.clientWidth / canvas.clientHeight;
	mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, aspect, 0.1, cameraDistance * 2.0);

	let viewMatrix = mat4.create();
	mat4.translate(viewMatrix, viewMatrix, vec3.fromValues(0, 0, -cameraDistance));
	let now = Date.now() / 1000;

	mat4.rotate(
		viewMatrix,
		viewMatrix,
		now,
		vec3.fromValues(0, 1, 0)
	);

	let modelViewProjectionMatrix = mat4.create();
	mat4.multiply(modelViewProjectionMatrix, projectionMatrix, viewMatrix);

	return modelViewProjectionMatrix;
}


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

		let state = this.getState(node);

		let {uniformBuffer, pipeline, uniformBindGroup} = state;

		const transformationMatrix = getTransformationMatrix();

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