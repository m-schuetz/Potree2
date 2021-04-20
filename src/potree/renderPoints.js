
import {Vector3, Matrix4} from "potree";
import {Timer} from "potree";


const vs = `
[[block]] struct Uniforms {
	[[size(64)]] worldView : mat4x4<f32>;
	[[size(64)]] proj : mat4x4<f32>;
	[[size(4)]] screen_width : f32;
	[[size(4)]] screen_height : f32;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

struct VertexInput {
	[[builtin(instance_index)]] instanceIdx : u32;
	[[location(0)]] position : vec4<f32>;
	[[location(1)]] color : vec4<f32>;
};

struct VertexOutput {
	[[builtin(position)]] position : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
};


[[stage(vertex)]]
fn main(vertex : VertexInput) -> VertexOutput {

	var output : VertexOutput;

	var viewPos : vec4<f32> = uniforms.worldView * vertex.position;
	output.position = uniforms.proj * viewPos;

	var c : vec4<f32> = vertex.color;

	// check if instance_index works
	// c.r = f32(vertex.instanceIdx);
	// c.g = 0.0;
	// c.b = 0.0;
	// c.a = 1.0;

	output.color = c;

	return output;
}
`;

const fs = `

struct FragmentInput {
	[[location(0)]] color : vec4<f32>;
};

struct FragmentOutput {
	[[location(0)]] color : vec4<f32>;
};

[[stage(fragment)]]
fn main(fragment : FragmentInput) -> FragmentOutput {
	var output : FragmentOutput;
	output.color = fragment.color;

	return output;
}
`;

let octreeStates = new Map();

function createPipeline(renderer){

	let {device} = renderer;

	const pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
			buffers: [
				{ // point position
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
			],
		},
		fragment: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'point-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: "less",
			format: "depth24plus-stencil8",
		},
	});

	return pipeline;
}

function getOctreeState(renderer, node){

	let {device} = renderer;

	let state = octreeStates.get(node);

	if(!state){
		let pipeline = createPipeline(renderer);

		const uniformBuffer = device.createBuffer({
			size: 2 * 4 * 16 + 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const uNodesBuffer = device.createBuffer({
			size: 16 * 1000,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		const uniformBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				// {binding: 1, resource: {buffer: uNodesBuffer}}
			],
		});

		state = {
			pipeline: pipeline,
			uniformBuffer: uniformBuffer,
			uniformBindGroup: uniformBindGroup,
			uNodesBuffer: uNodesBuffer,
		};

		octreeStates.set(node, state);

	}

	return state;
}

export function render(renderer, pass, octree, camera){

	let {device} = renderer;

	let octreeState = getOctreeState(renderer, octree);

	let nodes = octree.visibleNodes;

	{ // update uniforms
		

		{ // uniforms

			let {uniformBuffer} = octreeState;

			let data = new ArrayBuffer(256);
			let f32 = new Float32Array(data);
			let view = new DataView(data);

			let world = octree.world;
			let camView = camera.view;
			let worldView = new Matrix4().multiplyMatrices(camView, world);

			f32.set(worldView.elements, 0);
			f32.set(camera.proj.elements, 16);

			let size = renderer.getSize();

			view.setFloat32(128, size.width, true);
			view.setFloat32(132, size.height, true);

			renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, 136);
		}


		// { // nodes
		// 	let buffer = new Float32Array(4 * nodes.length);
		// 	for(let i = 0; i < nodes.length; i++){
		// 		buffer[4 * i + 0] = 0;
		// 		buffer[4 * i + 1] = i / 200;
		// 		buffer[4 * i + 2] = 0;
		// 		buffer[4 * i + 3] = 1;
		// 	}

		// 	device.queue.writeBuffer(
		// 		octreeState.uNodesBuffer, 0,
		// 		buffer.buffer, buffer.byteOffset, buffer.byteLength
		// 	);
		// }
	}

	let {passEncoder} = pass;
	let {pipeline, uniformBindGroup} = octreeState;

	Timer.timestamp(passEncoder, "points-start");

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, uniformBindGroup);

	let i = 0;
	for(let node of nodes){

		let vboPosition = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "position").buffer);
		let vboColor = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "rgba").buffer);

		passEncoder.setVertexBuffer(0, vboPosition);
		passEncoder.setVertexBuffer(1, vboColor);

		if(octree.showBoundingBox === true){
			let box = node.boundingBox.clone().applyMatrix4(octree.world);
			let position = box.min.clone();
			position.add(box.max).multiplyScalar(0.5);
			// position.applyMatrix4(octree.world);
			let size = box.size();
			// let color = new Vector3(...SPECTRAL.get(node.level / 5));
			let color = new Vector3(255, 255, 0);
			renderer.drawBoundingBox(position, size, color);
		}

		let numElements = node.geometry.numElements;
		passEncoder.draw(numElements, 1, 0, i);

		i++;
	}

	Timer.timestamp(passEncoder, "points-end");

};