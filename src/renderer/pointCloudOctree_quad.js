

let vs = `
#version 450

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldViewProj;
	ivec4 imin;
	vec4 offset;
	float screenWidth;
	float screenHeight;
} uniforms;

layout(location = 0) in ivec3 a_position;
layout(location = 1) in ivec4 a_rgb;

layout(location=2) in vec3 posBillboard;

layout(location = 0) out vec4 vColor;

void main() {
	vColor = vec4(
		float(a_rgb.x) / 256.0,
		float(a_rgb.y) / 256.0,
		float(a_rgb.z) / 256.0,
		1.0
	);

	ivec3 min = uniforms.imin.xyz;

	int ix = (a_position.x) / 1000;
	int iy = (a_position.y) / 1000;
	int iz = (a_position.z) / 1000;
	
	ix = ix / 1000;
	iy = iy / 1000;
	iz = iz / 1000;

	vec3 pos = vec3(
		float(ix) * 0.0031996278762817386,
		float(iy) * 0.004269749641418458,
		float(iz) * 0.004647889137268066
	);

	pos = pos + uniforms.offset.xyz;

	gl_Position = uniforms.worldViewProj * vec4(pos, 1.0);

	float w = gl_Position.w;
	float pointSize = 10.0;
	gl_Position.x += w * pointSize * posBillboard.x / uniforms.screenWidth;
	gl_Position.y += w * pointSize * posBillboard.y / uniforms.screenHeight;

	

}
`;


let fs = `

#version 450

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vColor;
	// outColor = vec4(1.0, 0.0, 0.0, 1.0);
}

`;

let shader = null;

let billboardBuffer = null;
function getBillboardBuffer(device){

	if(billboardBuffer === null){
		let values = [
			-1, -1, 0,
			1, -1, 0,
			1, 1, 0,
			-1, 1, 0
		];

		const [gpuBuffer, mapping] = device.createBufferMapped({
			size: values.length * 4,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		new Float32Array(mapping).set(values);
		gpuBuffer.unmap();

		billboardBuffer = gpuBuffer;

	}

	return billboardBuffer;
}

export function initializePointCloudOctreePipeline(octree){
	let {device} = this;

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
					stepMode: "instance",
					attributes: [
						{ // position
							shaderLocation: 0,
							offset: 0,
							format: "int3"
						}
					]
				},{
					arrayStride: 1 * 4,
					stepMode: "instance",
					attributes: [
						{ // color
							shaderLocation: 1,
							offset: 0,
							format: "uchar4"
						}
					]
				},{
					arrayStride: 4 * 4,
					attributes: [
						{ // billboard position
							shaderLocation: 2,
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
		primitiveTopology: 'triangle-strip',
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

	const uniformBufferSize = 4 * 16 
		+ 4 * 4 
		+ 4 * 4 
		+ 4 * 4;

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

	if(shader === null){
		shader = {
			vsModule: this.makeShaderModule('vertex', vs),
			fsModule: this.makeShaderModule('fragment', fs),
		};

		return;
	}

	if(!octree.webgpu){
		let {pipeline, bindGroupLayout} = initializePointCloudOctreePipeline.bind(this)(octree);
		let uniforms = initializePointCloudOctreeUniforms.bind(this)(octree, bindGroupLayout);

		octree.webgpu = {
			pipeline: pipeline,
			bindGroupLayout: bindGroupLayout,
			uniforms: uniforms,
		};
	}

	let {device} = this;
	let {webgpu} = octree;
	let {pipeline, uniforms} = webgpu;

	let transform = mat4.create();
	let scale = mat4.create();
	let translate = mat4.create();
	let worldView = mat4.create();
	let worldViewProj = mat4.create();
	let identity = mat4.create();

	passEncoder.setPipeline(pipeline);

	for(let node of octree.visibleNodes){
		if(!node.webgpu){
			let buffers = this.initializeBuffers(node);

			node.webgpu = {
				buffers: buffers,
			};
		}

		let webgpuNode = node.webgpu;
		let {buffers} = webgpuNode;

		mat4.scale(scale, identity, octree.scale.toArray());
		mat4.translate(translate, identity, octree.position.toArray());
		mat4.multiply(transform, translate, scale);

		mat4.multiply(worldView, view, transform);
		mat4.multiply(worldViewProj, proj, worldView);

		let [width, height] = [this.canvas.clientWidth, this.canvas.clientHeight];

		uniforms.buffer.setSubData(0, worldViewProj);
		uniforms.buffer.setSubData(4 * 16, new Int32Array([0, 0, 0, 0]));
		uniforms.buffer.setSubData(4 * 16 + 4 * 4, 
			new Float32Array([-0.748212993144989, -2.7804059982299805, 2.547821283340454, 0]));
		uniforms.buffer.setSubData(4 * 16 + 4 * 4 + 4 * 4, 
			new Float32Array([width, height]));

		

		let bufPos = buffers.find(b => b.name === "position");
		let bufCol = buffers.find(b => b.name === "rgb");
		passEncoder.setVertexBuffer(0, bufPos.handle);
		passEncoder.setVertexBuffer(1, bufCol.handle);
		passEncoder.setVertexBuffer(2, getBillboardBuffer(device));
		
		passEncoder.setBindGroup(0, uniforms.bindGroup);

		passEncoder.draw(4, node.numPoints, 0, 0);

	}

}