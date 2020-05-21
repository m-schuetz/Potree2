
let vs = `
	#version 450

	layout(set = 0, binding = 0) uniform Uniforms {
		mat4 worldViewProj;
	} uniforms;

	layout(location = 0) in vec3 a_position;
	layout(location = 1) in ivec4 a_rgb;

	layout(location = 0) out vec4 vColor;

	vec3 getColor(){
		vec3 rgb = vec3(a_rgb.xyz);
		if(length(rgb) > 2.0){
			rgb = rgb / 256.0;
		}
		if(length(rgb) > 2.0){
			rgb = rgb / 256.0;
		}
		return rgb;
	}

	void main() {
		vColor = vec4(getColor(), 1.0);

		gl_Position = uniforms.worldViewProj * vec4(a_position, 1.0);
	}
`;

let fs = `

	#version 450

	layout(location = 0) in vec4 vColor;
	layout(location = 0) out vec4 outColor;

	void main() {
		outColor = vColor;
		//outColor = vec4(vColor.xyz / 256.0, 1.0);
		//outColor = vec4(1.0, 0.0, 0.0, 1.0);
	}

`;



export function prepareStuff(renderer){

	let {device} = renderer;

	let shader = {
		vsModule: renderer.makeShaderModule('vertex', vs),
		fsModule: renderer.makeShaderModule('fragment', fs),
	};

	let bindGroupLayout = device.createBindGroupLayout({
		entries: [{
			binding: 0,
			visibility: GPUShaderStage.VERTEX,
			type: "uniform-buffer"
		}]
	});

	let pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

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
			vertexBuffers: [{
				arrayStride: 16,
				attributes: [{
					shaderLocation: 0,
					offset: 0,
					format: "float3",
				}],
			},{
				arrayStride: 16,
				attributes: [{
					shaderLocation: 1,
					offset: 12,
					format: "uchar4",
				}],
			}]
		},
		colorStates: [{
			format: renderer.swapChainFormat,
		}],
		primitiveTopology: 'point-list',
		depthStencilState: {
			depthWriteEnabled: true,
			depthCompare: "less",
			format: "depth24plus-stencil8",
		}
	});

	const uniformBufferSize = 64;

	let buffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	let bindGroup = device.createBindGroup({
		layout: bindGroupLayout,
		entries: [{
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

	return {
		shader: shader,
		bindGroupLayout: bindGroupLayout,
		pipeline: pipeline,
		bindGroup: bindGroup, 
		uniforms: uniforms,
	};
}

