
// ==========================================
// VERTEX SHADER
// ==========================================

let vs = `
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

let fs = `
#version 450

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vColor;
}
`;

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
								format: "float3"
							}
						]
					},{
						arrayStride: 4 * 4,
						attributes: [
							{ // color
								shaderLocation: 1,
								offset: 0,
								format: "float4"
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

}