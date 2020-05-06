
import {Geometry} from "../Geometry.js";

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










	let identity = mat4.create();
	let transform = mat4.create();
	let scale = mat4.create();
	let translate = mat4.create();
	let worldView = mat4.create();
	let worldViewProj = mat4.create();
	
	for(let i = 0; i < boxes.length; i++){
		let box = boxes[i];

		mat4.scale(scale, identity, box.scale.toArray());
		mat4.translate(translate, identity, box.position.toArray());
		mat4.multiply(transform, translate, scale);

		mat4.multiply(worldView, view, transform);
		mat4.multiply(worldViewProj, proj, worldView);

		let offset = i * 16 * 4;
		uniforms.buffer2.setSubData(offset, worldViewProj);
	}

	uniforms.buffer.setSubData(0, worldViewProj);

	return uniforms;


}

export function renderBoundingBoxes(boxes, view, proj, passEncoder){


	if(boxes.length === 0){
		return;
	}

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

	passEncoder.setPipeline(pipeline);

	for(let i = 0; i < gpuBuffers.length; i++){
		let buffer = gpuBuffers[i];
		passEncoder.setVertexBuffer(i, buffer.handle);
	}
	
	passEncoder.setBindGroup(0, uniforms.bindGroup);

	passEncoder.draw(geometry.numPrimitives, boxes.length, 0, 0);


}