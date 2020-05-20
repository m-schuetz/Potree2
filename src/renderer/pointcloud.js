import {Matrix4} from "../math/Matrix4.js";
import {toWebgpuAttribute, webgpuToGlsl} from "../octree/PointAttributes.js";

let vs = `
#version 450

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldViewProj;
	mat4 worldView;
	mat4 proj;
	float screenWidth;
	float screenHeight;
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

let shader = null;

let f32_for_mat4 = new Float32Array(4 * 4);
let i32_vec4 = new Int32Array(4);

export function initializePointCloudPipeline(octree){
	let {device} = this;

	let bindGroupLayout = device.createBindGroupLayout({
		entries: [{
			binding: 0,
			visibility: GPUShaderStage.VERTEX,
			type: "uniform-buffer"
		}]
	});

	let pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

	let vertexBuffers = [
		{
			arrayStride: 12,
			attributes: [{
				shaderLocation: 0,
				offset: 0,
				format: "float3",
			}],
		},{
			arrayStride: 4,
			attributes: [{
				shaderLocation: 1,
				offset: 0,
				format: "uchar4",
			}],
		}
	];

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
		vertexState: { vertexBuffers: vertexBuffers },
		colorStates: [{
			format: this.swapChainFormat,
		}],
		primitiveTopology: 'point-list',
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

export function initializePointCloudUniforms(octree, bindGroupLayout){
	let {device} = this;

	const uniformBufferSize = 3 * 64 + 2 * 16;

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

let testBuffer = null;
function getTestBuffer(size){
	if(testBuffer === null){
		let buffer = this.device.createBuffer({
			size: size,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
		});

		testBuffer = buffer;
	}

	return testBuffer;
}

export function renderPointCloud(pointcloud, view, proj, state){

	let {device} = this;

	if(shader === null){
		shader = {
			vsModule: this.makeShaderModule('vertex', vs),
			fsModule: this.makeShaderModule('fragment', fs),
		};

		return;
	}

	if(!pointcloud.webgpu){
		pointcloud.webgpu = {};
	}

	if(!pointcloud.webgpu.pipeline){
		let {pipeline, bindGroupLayout} = initializePointCloudPipeline.bind(this)(pointcloud);
		let uniforms = initializePointCloudUniforms.bind(this)(pointcloud, bindGroupLayout);

		pointcloud.webgpu.pipeline = pipeline;
		pointcloud.webgpu.bindGroupLayout = bindGroupLayout;
		pointcloud.webgpu.uniforms = uniforms;
	}

	

	if(!pointcloud.webgpu.buffers){
		let buffers = this.initializeBuffers(pointcloud.geometry);
		pointcloud.webgpu.buffers = buffers;
	}

	let {webgpu} = pointcloud;
	let {pipeline, uniforms, buffers} = webgpu;

	let transform = new Matrix4();
	let scale = new Matrix4();
	let translate = new Matrix4();
	let worldView = new Matrix4();
	let worldViewProj = new Matrix4();

	let commandEncoder = device.createCommandEncoder();

	// let testBuffer = getTestBuffer.bind(this)(pointcloud.numPoints * 16);

	// let n = Math.min(20_000_000, pointcloud.geometry.numPrimitives);
	// let steps = 10;
	// for(let i = 0; i < steps; i++){

	// 	let start = parseInt((i * n) / steps);
	// 	let size = parseInt(n / steps);

	// 	commandEncoder.copyBufferToBuffer(
	// 		buffers[0].handle, start * 12, 
	// 		testBuffer, start * 12,
	// 		size * 12
	// 	);
	// 	commandEncoder.copyBufferToBuffer(
	// 		buffers[1].handle, start * 4, 
	// 		testBuffer, n * 12 + start * 4,
	// 		size * 4
	// 	);
	// }
	// commandEncoder.copyBufferToBuffer(
	// 	buffers[0].handle, 0, 
	// 	testBuffer, 0,
	// 	n * 12
	// );
	// commandEncoder.copyBufferToBuffer(
	// 	buffers[1].handle, 0, 
	// 	testBuffer, n * 12,
	// 	n * 4
	// );

	device.defaultQueue.submit([commandEncoder.finish()]);
	commandEncoder = device.createCommandEncoder();

	let passEncoder = commandEncoder.beginRenderPass(state.renderPassDescriptor);
	passEncoder.setPipeline(pipeline);

	{
		scale.makeScale(pointcloud.scale.x, pointcloud.scale.y, pointcloud.scale.z);
		translate.makeTranslation(pointcloud.position.x, pointcloud.position.y, pointcloud.position.z);
		transform.multiplyMatrices(translate, scale);
		worldView.multiplyMatrices(view, transform);
		worldViewProj.multiplyMatrices(proj, worldView);

		let [width, height] = [this.canvas.clientWidth, this.canvas.clientHeight];

		f32_for_mat4.set(worldViewProj.elements)
		uniforms.buffer.setSubData(0, f32_for_mat4);
		f32_for_mat4.set(worldView.elements)
		uniforms.buffer.setSubData(64 + 0, f32_for_mat4);
		f32_for_mat4.set(proj.elements)
		uniforms.buffer.setSubData(128 + 0, f32_for_mat4);

		let i = 0;
		for(let buffer of buffers){
			passEncoder.setVertexBuffer(i, buffer.handle);

			i++;
		}

		passEncoder.setBindGroup(0, uniforms.bindGroup);

		passEncoder.draw(pointcloud.geometry.numPrimitives, 1, 0, 0);


		// passEncoder.setVertexBuffer(0, testBuffer, 0);
		// passEncoder.setVertexBuffer(1, testBuffer, n * 12);

		// passEncoder.draw(n, 1, 0, 0);
	}
		
	passEncoder.endPass();

	device.defaultQueue.submit([commandEncoder.finish()]);


}