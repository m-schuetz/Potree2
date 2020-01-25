
// ==========================================
// VERTEX SHADER
// ==========================================


let vsTest = `
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

let fsTest = `
#version 450

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vColor;
}
`;


class TestObject{

	constructor(){
		this.uploaded = false;
	}

	upload(renderer){

		let {device} = renderer;
		let {numPoints, positions, colors} = this;

		let bufPositions = device.createBuffer({
			size: 12 * numPoints,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
		});
		bufPositions.setSubData(0, positions);

		let bufColors = device.createBuffer({
			size: 16 * numPoints,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
		});
		bufColors.setSubData(0, colors);

		this.bufPositions = bufPositions;
		this.bufColors = bufColors;
	}

	createShader(renderer){
		let shader = {
			vsModule: renderer.createShader('vertex', vsTest),
			fsModule: renderer.createShader('fragment', fsTest),
		};

		this.shader = shader;
	}

	createPipeline(renderer){
		let {device, swapChainFormat} = renderer;
		//let shader = renderer.basicShader;
		let {shader} = this;

		const uniformsBindGroupLayout = device.createBindGroupLayout({
			bindings: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX,
				type: "uniform-buffer"
			}]
		});

		const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [uniformsBindGroupLayout] });

		const pipeline = device.createRenderPipeline({
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
					format: swapChainFormat,
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

		const uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const uniformBindGroup = device.createBindGroup({
			layout: uniformsBindGroupLayout,
			bindings: [{
				binding: 0,
				resource: {
					buffer: uniformBuffer,
				},
			}],
		});

		this.uniformsBindGroupLayout = uniformsBindGroupLayout;
		this.pipeline = pipeline;
		this.uniformBuffer = uniformBuffer;
		this.uniformBindGroup = uniformBindGroup;
	}

	render(renderer, timestamp){

		if(!this.uploaded){
			this.upload(renderer);
			this.createShader(renderer);
			this.createPipeline(renderer);

			this.uploaded = true;
		}

		let {canvas, device, swapChain, depthTexture} = renderer;
		let {pipeline, uniformBuffer, uniformBindGroup} = this;

		let worldViewProj = mat4.create();
		{ // update worldViewProj
			let proj = mat4.create();
			let view = mat4.create();

			{ // proj
				const aspect = Math.abs(canvas.width / canvas.height);
				mat4.perspective(proj, 45, aspect, 0.1, 20000.0);
			}

			{ // view
				let target = vec3.fromValues(0, 0, 0);
				let r = 5;
				let z = 3;
				let x = r * Math.sin(timestamp / 1000) + target[0];
				let y = r * Math.cos(timestamp / 1000) + target[1];

				let position = vec3.fromValues(x, y, z);
				let up = vec3.fromValues(0, 0, 1);
				mat4.lookAt(view, position, target, up);
			}

			mat4.multiply(worldViewProj, proj, view);
		}



		this.uniformBuffer.setSubData(0, worldViewProj);

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
		passEncoder.setPipeline(pipeline);

		{
			let {bufPositions, bufColors, numPoints} = this;

			passEncoder.setVertexBuffer(0, bufPositions);
			passEncoder.setVertexBuffer(1, bufColors);
			passEncoder.setBindGroup(0, uniformBindGroup);
			passEncoder.setViewport(0, 0, canvas.width, canvas.height, 0, 1);
			passEncoder.draw(numPoints, 1, 0, 0);
		}

		passEncoder.endPass();
		device.defaultQueue.submit([commandEncoder.finish()]);


	}

}


export function createTestData(n){

	let object = new TestObject();

	object.numPoints = n;

	{
		let positions = new Float32Array(3 * n);
		for(let i = 0; i < n; i++){
			positions[3 * i + 0] = 2.0 * Math.random() - 1.0;
			positions[3 * i + 1] = 2.0 * Math.random() - 1.0;
			positions[3 * i + 2] = 2.0 * Math.random() - 1.0;
		}
		object.positions = positions;
	}

	{
		let colors = new Float32Array(4 * n);
		for(let i = 0; i < n; i++){
			colors[4 * i + 0] = Math.random();
			colors[4 * i + 1] = Math.random();
			colors[4 * i + 2] = Math.random();
			colors[4 * i + 3] = 1.0;
		}
		object.colors = colors;
	}

	return object;
}