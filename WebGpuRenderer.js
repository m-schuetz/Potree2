
// ==========================================
// VERTEX SHADER
// ==========================================

let vs = `
#version 450

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldViewProj;
} uniforms;

layout(location = 0) in ivec3 position;
layout(location = 1) in ivec4 color;

layout(location = 0) out vec4 vColor;

void main() {
	//vColor = vec4(color.xyz, 1.0);
	vColor = vec4(
		float(color.x) / 256.0,
		float(color.y) / 256.0,
		float(color.z) / 256.0,
		1.0
	);

	ivec3 min = ivec3(41650162, 55830631, 225668106);

	int ix = (position.x - min.x) / 1000;
	int iy = (position.y - min.y) / 1000;
	int iz = (position.z - min.z) / 1000;
	
	ix = ix / 1000;
	iy = iy / 1000;
	iz = iz / 1000;

	vec3 pos = vec3(
		float(ix) * 0.0031996278762817386,
		float(iy) * 0.004269749641418458,
		float(iz) * 0.004647889137268066
	);

	gl_Position = uniforms.worldViewProj * vec4(pos, 1.0);
}
`;

// ==========================================
// FRAGMENT SHADER
// ==========================================

let fs = `
#version 450

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vColor;
	// outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
`;

import {vsMesh, fsMesh} from "./shaders.js";

function configureSwapChain(device, swapChainFormat, context) {
	const swapChainDescriptor = {
		device: device,
		format: swapChainFormat
	};

	return context.configureSwapChain(swapChainDescriptor);
}

function makeShaderModule_GLSL(glslang, device, type, source) {
	let shaderModuleDescriptor = {
		code: glslang.compileGLSL(source, type),
		source: source
	};

	let shaderModule = device.createShaderModule(shaderModuleDescriptor);
	return shaderModule;
}


export class WebGpuRenderer{

	constructor(canvas){
		this.canvas = canvas;
		this.swapChainFormat = "bgra8unorm";

		// this.init();
	}

	static async create(canvas){
		let renderer = new WebGpuRenderer(canvas);
		await renderer.init();

		return renderer;
	}

	async init(){
		// TODO: use local version
		let glslangModule = await import("https://unpkg.com/@webgpu/glslang@0.0.9/dist/web-devel/glslang.js");
		let glslang = await glslangModule.default();

		let gpu = navigator['gpu'];
		let adapter = await gpu.requestAdapter();
		let device = await adapter.requestDevice();
		let context = this.canvas.getContext('gpupresent');
		let swapChain = configureSwapChain(device, this.swapChainFormat, context);

		let shader = {
			vsModule: makeShaderModule_GLSL(glslang, device, 'vertex', vs),
			fsModule: makeShaderModule_GLSL(glslang, device, 'fragment', fs),
		};

		let shaderMesh = {
			vsModule: makeShaderModule_GLSL(glslang, device, 'vertex', vsMesh),
			fsModule: makeShaderModule_GLSL(glslang, device, 'fragment', fsMesh),
		};

		let uniformsBindGroupLayout = device.createBindGroupLayout({
			bindings: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX,
				type: "uniform-buffer"
			}]
		});

		let pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [uniformsBindGroupLayout] });

		let pipeline = device.createRenderPipeline({
			layout: pipelineLayout,
			vertexStage: {
				module: shader.vsModule,
				entryPoint: 'main'
			},
			fragmentStage: {
				module: shader.fsModule,
				entryPoint: 'main'
			},
			vertexState: {
				vertexBuffers: [
					{
						arrayStride: 3 * 4,
						attributes: [
							{ // position
								shaderLocation: 0,
								offset: 0,
								format: "int3"
							}
						]
					},{
						arrayStride: 1 * 4,
						attributes: [
							{ // color
								shaderLocation: 1,
								offset: 0,
								format: "uchar4"
							}
						]
					}
				]
			},
			colorStates: [
				{
					format: this.swapChainFormat,
					alphaBlend: {
						srcFactor: "src-alpha",
						dstFactor: "one-minus-src-alpha",
						operation: "add"
					}
				}
			],
			primitiveTopology: 'point-list',
			rasterizationState: {
				frontFace: "ccw",
				cullMode: 'none'
			},
			depthStencilState: {
				depthWriteEnabled: true,
				depthCompare: "less",
				format: "depth24plus-stencil8",
			}
		});

		const uniformBufferSize = 4 * 16; // 4x4 matrix

		let uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let uniformBindGroup = device.createBindGroup({
			layout: uniformsBindGroupLayout,
			bindings: [{
				binding: 0,
				resource: {
					buffer: uniformBuffer,
				},
			}],
		});

		let depthTexture = device.createTexture({
			size: {
				width: canvas.width,
				height: canvas.height,
				depth: 1
			},
			format: "depth24plus-stencil8",
			usage: GPUTextureUsage.OUTPUT_ATTACHMENT
		});

		this.gpu = gpu;
		this.adapter = adapter;
		this.device = device;
		this.context = context;
		this.swapChain = swapChain;
		this.pipeline = pipeline;

		this.shader = shader;
		this.shaderMesh = shaderMesh;
		this.uniformBuffer = uniformBuffer;
		this.uniformBindGroup = uniformBindGroup;
		this.depthTexture = depthTexture;
	}

	configureSwapChain(device, swapChainFormat, context){
		const swapChainDescriptor = {
			device: device,
			format: swapChainFormat
		};

		return context.configureSwapChain(swapChainDescriptor);
	}










	initializeMeshBuffers(mesh){
		let {device, shaderMesh} = this;

		let {n, position, color} = mesh;
		
		let [bufPositions, posMapping] = device.createBufferMapped({
			size: 12 * n,
			usage: GPUBufferUsage.VERTEX,
		});
		new Float32Array(posMapping).set(new Float32Array(position));
		bufPositions.unmap();

		let [bufRGBA, mappingRGB] = device.createBufferMapped({
			size: 4 * n,
			usage: GPUBufferUsage.VERTEX,
		});
		new Uint8Array(mappingRGB).set(new Uint8Array(color));
		bufRGBA.unmap();

		let buffers = {
			position: bufPositions,
			color: bufRGBA,
		};

		return buffers;
	}

	initializeMeshPipeline(mesh){
		let {device, shaderMesh} = this;

		let bindGroupLayout = device.createBindGroupLayout({
			bindings: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX,
				type: "uniform-buffer"
			}]
		});

		let pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
		let shader = shaderMesh;

		let pipeline = device.createRenderPipeline({
			layout: pipelineLayout,
			vertexStage: {
				module: shader.vsModule,
				entryPoint: 'main'
			},
			fragmentStage: {
				module: shader.fsModule,
				entryPoint: 'main'
			},
			vertexState: {
				vertexBuffers: [
					{
						arrayStride: 3 * 4,
						attributes: [
							{ // position
								shaderLocation: 0,
								offset: 0,
								format: "float3"
							}
						]
					},{
						arrayStride: 1 * 4,
						attributes: [
							{ // color
								shaderLocation: 1,
								offset: 0,
								format: "uchar4"
							}
						]
					}
				]
			},
			colorStates: [
				{
					format: this.swapChainFormat,
					alphaBlend: {
						srcFactor: "src-alpha",
						dstFactor: "one-minus-src-alpha",
						operation: "add"
					}
				}
			],
			primitiveTopology: 'triangle-list',
			rasterizationState: {
				frontFace: "ccw",
				cullMode: 'none'
			},
			depthStencilState: {
				depthWriteEnabled: true,
				depthCompare: "less",
				format: "depth24plus-stencil8",
			}
		});

		return {
			pipeline: pipeline,
			bindGroupLayout: bindGroupLayout,
		};
	}

	initializeMeshUniforms(mesh, bindGroupLayout){
		let {device, shaderMesh} = this;

		const uniformBufferSize = 4 * 16; // 4x4 matrix

		let buffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let bindGroup = device.createBindGroup({
			layout: bindGroupLayout,
			bindings: [{
				binding: 0,
				resource: {
					buffer: buffer,
				},
			}],
		});

		let uniforms = {
			buffer: buffer,
			bindGroup: bindGroup,
			bindGroupLayout: bindGroupLayout,
		};

		return uniforms;
	}

	initializeMesh(mesh){

		let buffers = this.initializeMeshBuffers(mesh);
		let {pipeline, bindGroupLayout} = this.initializeMeshPipeline(mesh);
		let uniforms = this.initializeMeshUniforms(mesh, bindGroupLayout);

		let webgpu = {
			buffers: buffers,
			pipeline: pipeline,
			uniforms: uniforms,
		};

		mesh.webgpu = webgpu;

	}

	renderMesh(mesh, worldViewProj, passEncoder){

		if(!mesh.webgpu){
			this.initializeMesh(mesh);
		}

		let {buffers, pipeline, uniforms} = mesh.webgpu;

		uniforms.buffer.setSubData(0, worldViewProj);

		passEncoder.setPipeline(pipeline);

		passEncoder.setVertexBuffer(0, buffers.position);
		passEncoder.setVertexBuffer(1, buffers.color);
		passEncoder.setBindGroup(0, uniforms.bindGroup);

		passEncoder.draw(mesh.n, 1, 0, 0);
	}

	renderObject(object, passEncoder){

		if(!object.webgpu){
			this.setupObject(object);
		}

		let {buffers, pipeline, uniforms} = object.webgpu;

		uniforms.buffer.setSubData(0, worldViewProj);

		passEncoder.setPipeline(pipeline);

		passEncoder.setVertexBuffer(0, buffers.position);
		passEncoder.setVertexBuffer(1, buffers.color);
		passEncoder.setBindGroup(0, uniforms.bindGroup);

		passEncoder.draw(object.n, 1, 0, 0);

	}

	resize(){
		let {canvas, device} = this;

		let needsResize = canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight;
		if(needsResize){
			canvas.width = canvas.clientWidth;
			canvas.height = canvas.clientHeight;

			this.depthTexture = device.createTexture({
				size: {
					width: canvas.width,
					height: canvas.height,
					depth: 1
				},
				format: "depth24plus-stencil8",
				usage: GPUTextureUsage.OUTPUT_ATTACHMENT
			});
		}
	}

	render(objects, camera, dbg){

		
		this.resize();

		let aspect = this.canvas.width / this.canvas.height;
		let view = camera.getView();
		let proj = camera.getProjection(aspect);
		let worldViewProj = mat4.create();
		mat4.multiply(worldViewProj, proj, view);

		this.uniformBuffer.setSubData(0, worldViewProj);

		let {device, swapChain, depthTexture} = this;
		const commandEncoder = device.createCommandEncoder();
		const textureView = swapChain.getCurrentTexture().createView();
		const renderPassDescriptor = {
			colorAttachments: [{
				attachment: textureView,
				loadValue: { r: 0, g: 0, b: 0, a: 0 },
			}],
			depthStencilAttachment: {
				attachment: depthTexture.createView(),
				depthLoadValue: 1.0,
				depthStoreOp: "store",
				stencilLoadValue: 0,
				stencilStoreOp: "store",
			}
		};
		const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

		passEncoder.setViewport(0, 0, canvas.width, canvas.height, 0, 1);

		if(dbg){
			passEncoder.setPipeline(this.pipeline);
			passEncoder.setVertexBuffer(0, dbg.bufPositions);
			passEncoder.setVertexBuffer(1, dbg.bufColors);
			passEncoder.setBindGroup(0, this.uniformBindGroup);
			passEncoder.draw(dbg.n, 1, 0, 0);
		}

		for(let object of objects){
			// passEncoder.setPipeline(object.webgpu.pipeline);
			this.renderMesh(object, worldViewProj, passEncoder);
		}

		passEncoder.endPass();
		device.defaultQueue.submit([commandEncoder.finish()]);



		counter++;

		if(counter === 100){
			console.log("view", view);
			console.log("proj", proj);
		}

	}

}

let counter = 0;