
import {Vector3, Matrix4} from "../../math/math.js";

const vs = `
[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> in_position : vec4<f32>;
[[location(1)]] var<in> in_normal : vec4<f32>;

[[builtin(position)]] var<out> Position : vec4<f32>;

[[location(0)]] var<out> out_color : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {

	Position = uniforms.proj * uniforms.worldView * in_position;

	out_color = vec4<f32>(in_normal.xyz, 1.0);

	return;
}
`;

const fs = `

[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> in_color : vec4<f32>;

[[location(0)]] var<out> out_color : vec4<f32>;

[[stage(fragment)]]
fn main() -> void {

	// var N : vec3<f32> = (uniforms.worldView * vec4<f32>(in_normal.xyz, 0.0)).xyz;
	// N = normalize(N);

	out_color = in_color;

	return;
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
		vertexStage: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
		},
		fragmentStage: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
		},
		primitiveTopology: "triangle-list",
		depthStencilState: {
			depthWriteEnabled: true,
			depthCompare: "less",
			format: "depth24plus-stencil8",
		},
		vertexState: {
			vertexBuffers: [
				{ // position
					arrayStride: 3 * 4,
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float3",
					}],
				},{ // normal
					arrayStride: 3 * 4,
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "float3",
					}],
				},
			],
		},
		rasterizationState: {
			cullMode: "none",
		},
		colorStates: [{
				format: "bgra8unorm",
		}],
	});

	const uniformBufferSize = 256; 

	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

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
		device.defaultQueue.writeBuffer(
			uniformBuffer, 0,
			tmp.buffer, tmp.byteOffset, tmp.byteLength
		);

		tmp.set(camera.proj.elements);
		device.defaultQueue.writeBuffer(
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

	passEncoder.setVertexBuffer(0, vbos[0].vbo);
	passEncoder.setVertexBuffer(1, vbos[3].vbo);

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