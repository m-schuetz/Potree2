
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

import {vsMesh, fsMesh} from "../../shaders.js";

import {renderMesh} from "./mesh.js";
import {renderLines} from "./lines.js";
import {renderPointCloudOctree} from "./pointCloudOctree.js";

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
		this.geometryBuffers = new Map();

		this.nodeRenderers = {
			"Mesh": renderMesh.bind(this),
			"Lines": renderLines.bind(this),
			"PointCloudOctree": renderPointCloudOctree.bind(this),
		};

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

		this.shader = shader;
		this.shaderMesh = shaderMesh;
		this.depthTexture = depthTexture;
	}

	configureSwapChain(device, swapChainFormat, context){
		const swapChainDescriptor = {
			device: device,
			format: swapChainFormat
		};

		return context.configureSwapChain(swapChainDescriptor);
	}










	initializeBuffers(geometry){
		let {device} = this;

		if(this.geometryBuffers.has(geometry)){
			let gpuBuffers = this.geometryBuffers.get(geometry);

			return gpuBuffers;
		}else{
			let {numPrimitives, buffers} = geometry;

			let gpuBuffers = [];
			for(let buffer of buffers){

				let XArray = buffer.array.constructor;
				
				let [gpuBuffer, mapping] = device.createBufferMapped({
					size: buffer.array.byteLength,
					usage: GPUBufferUsage.VERTEX,
				});
				new XArray(mapping).set(buffer.array);
				gpuBuffer.unmap();

				gpuBuffers.push({name: buffer.name, handle: gpuBuffer});
			}

			this.geometryBuffers.set(geometry, gpuBuffers);

			return gpuBuffers;
		}
	}


	renderNode(node, view, proj, passEncoder){

		let nodeRenderer = this.nodeRenderers[node.constructor.name];
		
		if(nodeRenderer){
			nodeRenderer(node, view, proj, passEncoder);
		}else{
			//console.log(`no renderer found for: ${node.constructor.name}`);
		}

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

	createEncoders(){
		let {device, swapChain, depthTexture} = this;

		let commandEncoder = device.createCommandEncoder();
		let textureView = swapChain.getCurrentTexture().createView();
		let renderPassDescriptor = {
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
		
		let passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

		return [commandEncoder, passEncoder];
	}

	getVisibleNodes(scene, camera){

		let nodes = [];
		let stack = [scene.root];
		while(stack.length > 0){
			let node = stack.pop();

			nodes.push(node);

			for(let child of node.children){
				stack.push(child);
			}
		}

		return nodes;
	}

	render(scene, camera){

		let nodes = this.getVisibleNodes(scene, camera);

		this.resize();

		let aspect = this.canvas.width / this.canvas.height;
		let view = camera.getView();
		let proj = camera.getProjection(aspect);
		let worldViewProj = mat4.create();
		mat4.multiply(worldViewProj, proj, view);

		let [commandEncoder, passEncoder] = this.createEncoders();

		passEncoder.setViewport(0, 0, canvas.width, canvas.height, 0, 1);

		for(let node of nodes){
			this.renderNode(node, view, proj, passEncoder);
		}

		passEncoder.endPass();
		this.device.defaultQueue.submit([commandEncoder.finish()]);


	}

}
