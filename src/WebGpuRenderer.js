

// ==========================================
// VERTEX SHADER
// ==========================================

let vsBasic = `
#version 450

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldViewProj;
} uniforms;

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 vColor;

void main() {
	vColor = color;
	gl_Position = uniforms.worldViewProj * vec4(position, 1.0);
}
`;

// ==========================================
// FRAGMENT SHADER
// ==========================================

let fsBasic = `
#version 450

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vColor;
}
`;



function createSwapChain(device, format, context) {
	const descriptor = {
		device: device,
		format: format
	};

	return context.configureSwapChain(descriptor);
}

export class WebGpuRenderer{

	constructor(canvas){
		this.canvas = canvas;
		this.context = null;
		this.gpu = null;
		this.adapted = null;
		this.device = null;
		this.swapChain = null;
		this.glslangModule = null;
		this.glslang = null;

		this.basicShader = null;

		this.depthTexture = null;
	}

	static async create(canvas){
		let renderer = new WebGpuRenderer(canvas);

		await renderer.init();

		return renderer;
	}

	async init(){
		let gpu = navigator['gpu'];
		let glslangModule = await import("../libs/glslang/glslang.js");
		let glslang = await glslangModule.default();
		let adapter = await gpu.requestAdapter();
		let device = await adapter.requestDevice();
		let context = this.canvas.getContext('gpupresent')

		const swapChainFormat = "bgra8unorm";
		let swapChain = createSwapChain(
			device, swapChainFormat, context);

		let depthTexture = device.createTexture({
			size: {
				width: this.canvas.width,
				height: this.canvas.height,
				depth: 1
			},
			format: "depth24plus-stencil8",
			usage: GPUTextureUsage.OUTPUT_ATTACHMENT
		});

		this.gpu = gpu;
		this.glslangModule = glslangModule;
		this.glslang = glslang;
		this.adapter = adapter;
		this.device = device;
		this.context = context;
		this.swapChainFormat = swapChainFormat;
		this.swapChain = swapChain;
		this.depthTexture = depthTexture;

		let basicShader = {
			vsModule: this.createShader('vertex', vsBasic),
			fsModule: this.createShader('fragment', fsBasic),
		};

		this.basicShader = basicShader;
	}

	createShader(type, source) {
		let {glslang, device} = this;

		let descriptor = {
			code: glslang.compileGLSL(source, type),
			source: source
		};

		let shaderModule = device.createShaderModule(descriptor);

		return shaderModule;
	}

	render(){
		let {canvas, device, depthTexture} = this;

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
	
}