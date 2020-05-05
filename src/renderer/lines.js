
export function initializeLinesPipeline(mesh){
	let {device, shaderMesh} = this;

	let bindGroupLayout = device.createBindGroupLayout({
		entries: [{
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
		primitiveTopology: 'line-list',
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

export function initializeLinesUniforms(lines, bindGroupLayout){
	let {device} = this;

	const uniformBufferSize = 4 * 16; // 4x4 matrix

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

	return uniforms;
}

export function initializeLines(lines){

	let buffers = this.initializeBuffers(lines.geometry);
	let {pipeline, bindGroupLayout} = initializeLinesPipeline.bind(this)(lines);
	let uniforms = initializeLinesUniforms.bind(this)(lines, bindGroupLayout);

	let webgpu = {
		buffers: buffers,
		pipeline: pipeline,
		uniforms: uniforms,
	};

	lines.webgpu = webgpu;

}

export function renderLines(lines, view, proj, passEncoder){

	if(!lines.webgpu){
		initializeLines.bind(this)(lines);
	}

	let {buffers, pipeline, uniforms} = lines.webgpu;

	let transform = mat4.create();
	let scale = mat4.create();
	mat4.scale(scale, scale, lines.scale.toArray());
	let translate = mat4.create();
	mat4.translate(translate, translate, lines.position.toArray());
	mat4.multiply(transform, translate, scale);

	let worldView = mat4.create();
	mat4.multiply(worldView, view, transform);

	let worldViewProj = mat4.create();
	mat4.multiply(worldViewProj, proj, worldView);

	uniforms.buffer.setSubData(0, worldViewProj);

	passEncoder.setPipeline(pipeline);

	for(let i = 0; i < buffers.length; i++){
		let buffer = buffers[i];
		passEncoder.setVertexBuffer(i, buffer.handle);
	}
	
	passEncoder.setBindGroup(0, uniforms.bindGroup);

	passEncoder.draw(lines.geometry.numPrimitives, 1, 0, 0);
}