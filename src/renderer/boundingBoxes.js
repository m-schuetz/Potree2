
import {Geometry} from "../Geometry.js";
import {Matrix4} from "../math/Matrix4.js";

export let vs = `
#version 450

layout(location = 0) in vec3 position;

layout(location = 0) out vec4 vColor;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldViewProj;
} uniforms;

layout(set = 0, binding = 1) uniform Uniforms1 {
	mat4 worldViewProj[200];
} uniforms1;

void main() {
	vColor = vec4(1, 0, 0, 1);

	mat4 transform = uniforms1.worldViewProj[gl_InstanceIndex];
	gl_Position = transform * vec4(position, 1.0);

}
`;

export let fs = `
#version 450

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vColor;
}
`;

let shader = null;
let gpuBuffers = null;
let geometry = Geometry.createBoundingBox();

function createPipeline(){

	let {device, swapChainFormat} = this;

	let bindGroupLayout = device.createBindGroupLayout({
		entries: [{
			binding: 0,
			visibility: GPUShaderStage.VERTEX,
			type: "uniform-buffer"
		},{
			binding: 1,
			visibility: GPUShaderStage.VERTEX,
			type: "uniform-buffer"
		}]
	});

	let pipeline = device.createRenderPipeline({
		layout: device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
		vertexStage: {
			module: shader.vsModule,
			entryPoint: "main",
		},
		fragmentStage: {
			module: shader.fsModule,
			entryPoint: "main",
		},
		vertexState: {
			vertexBuffers: [
				{
					arrayStride: 3 * 4,
					attributes: [{
						shaderLocation: 0,
						offset: 0,
						format: "float3",
					}]
				}
			]
		},
		primitiveTopology: "line-list",
		colorStates: [{
			format: this.swapChainFormat,
			alphaBlend: {
				srcFactor: "src-alpha",
				dstFactor: "one-minus-src-alpha",
				operation: "add"
			}
		}],
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

function createUniforms(boxes, bindGroupLayout, view, proj){

	let {device} = this;

	const uniformBufferSize = 4 * 16; // 4x4 matrix

	let buffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	let buffer2 = device.createBuffer({
		size: boxes.length * uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	let bindGroup = device.createBindGroup({
		layout: bindGroupLayout,
		entries: [{
			binding: 0,
			resource: {
				buffer: buffer,
			},
		},{
			binding: 1,
			resource: {
				buffer: buffer2,
			},
		}],
	});

	let uniforms = {
		buffer: buffer,
		buffer2: buffer2,
		bindGroup: bindGroup,
		bindGroupLayout: bindGroupLayout,
	};

	let transform = new Matrix4();
	let scale = new Matrix4();
	let translate = new Matrix4();
	let worldView = new Matrix4();
	let worldViewProj = new Matrix4();
	
	for(let i = 0; i < boxes.length; i++){
		let box = boxes[i];

		scale.makeScale(box.scale.x, box.scale.y, box.scale.z);
		translate.makeTranslation(box.position.x, box.position.y, box.position.z);
		transform.multiplyMatrices(translate, scale);
		worldView.multiplyMatrices(view, transform);
		worldViewProj.multiplyMatrices(proj, worldView);

		let offset = i * 16 * 4;
		uniforms.buffer2.setSubData(offset, new Float32Array(worldViewProj.elements));
	}

	uniforms.buffer.setSubData(0, new Float32Array(worldViewProj.elements));

	return uniforms;


}

export function renderBoundingBoxes(boxes, view, proj, state){

	if(boxes.length === 0){
		return;
	}

	let {device} = this;

	if(shader === null){
		shader = {
			vsModule: this.makeShaderModule('vertex', vs),
			fsModule: this.makeShaderModule('fragment', fs),
		};
	}

	if(gpuBuffers === null){
		gpuBuffers = this.initializeBuffers(geometry);
	}

	let {pipeline, bindGroupLayout} = createPipeline.bind(this)();
	let uniforms = createUniforms.bind(this)(boxes, bindGroupLayout, view, proj);

	let commandEncoder = device.createCommandEncoder();

	let passEncoder = commandEncoder.beginRenderPass(state.renderPassDescriptor);
	passEncoder.setPipeline(pipeline);

	for(let i = 0; i < gpuBuffers.length; i++){
		let buffer = gpuBuffers[i];
		passEncoder.setVertexBuffer(i, buffer.handle);
	}
	
	passEncoder.setBindGroup(0, uniforms.bindGroup);

	passEncoder.draw(geometry.numPrimitives, boxes.length, 0, 0);
	passEncoder.endPass();

	device.defaultQueue.submit([commandEncoder.finish()]);

}