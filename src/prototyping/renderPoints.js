
import {Vector3, Matrix4} from "potree";
import {SceneNode} from "potree";
import * as Timer from "../renderer/Timer.js";

const vs = `
[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> pos_point : vec4<f32>;
[[location(1)]] var<in> color : vec4<f32>;

[[builtin(position)]] var<out> out_pos : vec4<f32>;
[[location(0)]] var<out> fragColor : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {

	var viewPos : vec4<f32> = uniforms.worldView * pos_point;
	out_pos = uniforms.proj * viewPos;

	var c : vec4<f32> = color;
	fragColor = c;

	return;
}
`;

const fs = `
[[location(0)]] var<in> fragColor : vec4<f32>;
[[location(0)]] var<out> outColor : vec4<f32>;

[[stage(fragment)]]
fn main() -> void {
	outColor = fragColor;

	return;
}
`;



let initialized = false;
let uniformBuffer = null;
let bindGroup = null;
let pipeline = null;

function init(renderer){

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
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				},{ // color
					arrayStride: 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "unorm8x4",
					}],
				},
			]
		},
		fragment: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'point-list',
			cullMode: 'back',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	uniformBuffer = device.createBuffer({
		size: 256,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
		],
	});

	


}



export function render(renderer, pass, node, camera){

	let {device} = renderer;

	init(renderer);

	{ // update uniforms
		let data = new ArrayBuffer(140);
		let f32 = new Float32Array(data);

		{ // transform
			let world = node.world;
			let view = camera.view;
			let worldView = new Matrix4().multiplyMatrices(view, world);

			f32.set(worldView.elements, 0);
			f32.set(camera.proj.elements, 16);
		}

		device.queue.writeBuffer(
			uniformBuffer, 0,
			data, 0, data.byteLength
		);
	}

	let {passEncoder} = pass;
	Timer.timestamp(passEncoder, "points-start");

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	let vboPosition = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "position").buffer);
	let vboColor = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "rgba").buffer);

	passEncoder.setVertexBuffer(0, vboPosition);
	passEncoder.setVertexBuffer(1, vboColor);

	let numElements = node.geometry.numElements;
	passEncoder.draw(numElements, 1, 0, 0);

	Timer.timestamp(passEncoder, "points-end");

}