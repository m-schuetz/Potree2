export function initializePointCloudOctreePipeline(octree){
	let {device, shader} = this;

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

	return {
		pipeline: pipeline,
		bindGroupLayout: bindGroupLayout,
	};
}

export function initializePointCloudOctreeUniforms(octree, bindGroupLayout){
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

export function renderPointCloudOctree(octree, view, proj, passEncoder){

	if(!octree.webgpu){
		let {pipeline, bindGroupLayout} = initializePointCloudOctreePipeline.bind(this)(octree);
		let uniforms = initializePointCloudOctreeUniforms.bind(this)(octree, bindGroupLayout);

		octree.webgpu = {
			pipeline: pipeline,
			bindGroupLayout: bindGroupLayout,
			uniforms: uniforms,
		};
	}

	let {webgpu} = octree;
	let {pipeline, uniforms} = webgpu;

	for(let node of octree.visibleNodes){
		if(!node.webgpu){
			let buffers = this.initializeBuffers(node);

			node.webgpu = {
				buffers: buffers,
			};
		}

		let webgpuNode = node.webgpu;
		let {buffers} = webgpuNode;

		let transform = mat4.create();
		let scale = mat4.create();
		mat4.scale(scale, scale, octree.scale.toArray());
		let translate = mat4.create();
		mat4.translate(translate, translate, octree.position.toArray());
		mat4.multiply(transform, translate, scale);

		let worldView = mat4.create();
		mat4.multiply(worldView, view, transform);

		let worldViewProj = mat4.create();
		mat4.multiply(worldViewProj, proj, worldView);

		uniforms.buffer.setSubData(0, worldViewProj);

		passEncoder.setPipeline(pipeline);

		// for(let i = 0; i < buffers.length; i++){
		// 	let buffer = buffers[i];
		// 	passEncoder.setVertexBuffer(i, buffer.handle);
		// }
		let bufPos = buffers.find(b => b.name === "position");
		let bufCol = buffers.find(b => b.name === "rgb");
		passEncoder.setVertexBuffer(0, bufPos.handle);
		passEncoder.setVertexBuffer(1, bufCol.handle);
		
		passEncoder.setBindGroup(0, uniforms.bindGroup);

		passEncoder.draw(node.numPoints, 1, 0, 0);

	}

}