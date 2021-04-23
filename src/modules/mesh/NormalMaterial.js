
import {Vector3, Matrix4} from "../../math/math.js";

const vs = `
[[block]] struct Uniforms {
	worldView : mat4x4<f32>;
	proj : mat4x4<f32>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> position : vec4<f32>;
[[location(1)]] var<in> normal : vec4<f32>;

struct VertexInput {
	[[location(0)]] position        : vec4<f32>;
	[[location(1)]] normal          : vec4<f32>;
};

struct VertexOutput {
	[[builtin(position)]] position  : vec4<f32>;
	[[location(3)]] color           : vec4<f32>;
};

[[stage(vertex)]]
fn main(vertex : VertexInput) -> VertexOutput {

	var output : VertexOutput;

	output.position = uniforms.proj * uniforms.worldView * vertex.position;

	output.color = vec4<f32>(vertex.normal.xyz, 1.0);

	return output;
}
`;

const fs = `

[[block]] struct Uniforms {
	worldView : mat4x4<f32>;
	proj : mat4x4<f32>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

struct FragmentInput {
	[[builtin(position)]] position  : vec4<f32>;
	[[location(3)]] color           : vec4<f32>;
};

struct FragmentOutput {
	[[location(0)]] color : vec4<f32>;
};

[[stage(fragment)]]
fn main(fragment : FragmentInput) -> FragmentOutput {

	// var N : vec3<f32> = (uniforms.worldView * vec4<f32>(in_normal.xyz, 0.0)).xyz;
	// N = normalize(N);

	var output : FragmentOutput;
	output.color = fragment.color;

	return output;
}
`;

let initialized = false;
let pipeline = null;
let uniformBuffer = null;

function initialize(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
			buffers: [
				{ // position
					arrayStride: 3 * 4,
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				},{ // normal
					arrayStride: 3 * 4,
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "float32x3",
					}],
				},
			],
		},
		fragment: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: false,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	const uniformBufferSize = 256; 

	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	initialized = true;
}

export function render(renderer, pass, node, camera, renderables){
	
	let {device} = renderer;

	initialize(renderer);

	{ // update uniforms
		let world = node.world;
		let view = camera.view;
		let worldView = new Matrix4().multiplyMatrices(view, world);

		let tmp = new Float32Array(16);

		tmp.set(worldView.elements);
		device.queue.writeBuffer(
			uniformBuffer, 0,
			tmp.buffer, tmp.byteOffset, tmp.byteLength
		);

		tmp.set(camera.proj.elements);
		device.queue.writeBuffer(
			uniformBuffer, 64,
			tmp.buffer, tmp.byteOffset, tmp.byteLength
		);
	}

	let {passEncoder} = pass;
	let vbos = renderer.getGpuBuffers(node.geometry);

	passEncoder.setPipeline(pipeline);

	let bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
		]
	});

	passEncoder.setBindGroup(0, bindGroup);

	let vboPosition = vbos.find(item => item.name === "position").vbo;
	let vboNormal = vbos.find(item => item.name === "normal").vbo;
	passEncoder.setVertexBuffer(0, vboPosition);
	passEncoder.setVertexBuffer(1, vboNormal);

	if(node.geometry.indices){
		let indexBuffer = renderer.getGpuBuffer(node.geometry.indices);

		passEncoder.setIndexBuffer(indexBuffer, "uint32", 0, indexBuffer.byteLength);

		let numIndices = node.geometry.indices.length;
		passEncoder.drawIndexed(numIndices);
	}else{
		let numElements = node.geometry.numElements;
		passEncoder.draw(numElements, 1, 0, 0);
	}

}

export class NormalMaterial{

	constructor(){

	}
	
}