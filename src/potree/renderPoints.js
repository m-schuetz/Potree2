
import {Vector3, Matrix4} from "../math/math.js";
import {SPECTRAL} from "../misc/Gradients.js";


const vs = `
[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
	[[offset(128)]] screen_width : f32;
	[[offset(132)]] screen_height : f32;
};

struct NodeBuffer{
	[[offset(0)]] color : vec4<f32>;
};

[[block]] struct Nodes{
	[[offset(0)]] values : [[stride(16)]] array<NodeBuffer, 1000>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), set(0)]] var<uniform> nodes : Nodes;

[[location(0)]] var<in> pos_point : vec4<f32>;
[[location(1)]] var<in> color : vec4<f32>;

[[builtin(instance_idx)]] var<in> instanceIdx : i32;
[[builtin(position)]] var<out> out_pos : vec4<f32>;
[[location(0)]] var<out> fragColor : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {

	var viewPos : vec4<f32> = uniforms.worldView * pos_point;
	var far : f32 = 10000.0;

	out_pos = uniforms.proj * viewPos;
	out_pos.z = -viewPos.z * out_pos.w / far;

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

let octreeStates = new Map();
let nodeStates = new Map();

function createPipeline(renderer){

	let {device} = renderer;

	const pipeline = device.createRenderPipeline({
		vertexStage: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
		},
		fragmentStage: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
		},
		primitiveTopology: "point-list",
		depthStencilState: {
			depthWriteEnabled: true,
			depthCompare: "less",
			format: "depth24plus-stencil8",
		},
		vertexState: {
			vertexBuffers: [
				{ // point position
					arrayStride: 3 * 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float3",
					}],
				},{ // color
					arrayStride: 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "uchar4norm",
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
			entries: [{
				binding: 0,
				resource: {buffer: uniformBuffer},
			},{
				binding: 1,
				resource: {buffer: uNodesBuffer},
			}],
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

function getNodeState(renderer, node){
	let state = nodeStates.get(node);

	if(!state){
		let vbos = renderer.getGpuBuffers(node.geometry);

		state = {vbos};
		nodeStates.set(node, state);
	}

	return state;
}

export function render(renderer, pass, octree, camera){

	let {device} = renderer;

	let octreeState = getOctreeState(renderer, octree);

	let nodes = octree.visibleNodes;

	{ // update uniforms
		let {uniformBuffer} = octreeState;

		{ // transform
			let world = octree.world;
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

		{ // screen size
			let size = renderer.getSize();
			let data = new Float32Array([size.width, size.height]);
			device.defaultQueue.writeBuffer(
				uniformBuffer,
				128,
				data.buffer,
				data.byteOffset,
				data.byteLength
			);
		}

		{ // nodes
			let buffer = new Float32Array(4 * nodes.length);
			for(let i = 0; i < nodes.length; i++){
				buffer[4 * i + 0] = 0;
				buffer[4 * i + 1] = i / 200;
				buffer[4 * i + 2] = 0;
				buffer[4 * i + 3] = 1;
			}

			device.defaultQueue.writeBuffer(
				octreeState.uNodesBuffer, 0,
				buffer.buffer, buffer.byteOffset, buffer.byteLength
			);

		}
	}

	let {passEncoder} = pass;
	let {pipeline, uniformBindGroup} = octreeState;

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, uniformBindGroup);

	for(let node of nodes){
		let nodeState = getNodeState(renderer, node);

		passEncoder.setVertexBuffer(0, nodeState.vbos[0].vbo);
		passEncoder.setVertexBuffer(1, nodeState.vbos[1].vbo);

		// if(octree.showBoundingBox === true){
		// 	let position = node.boundingBox.min.clone();
		// 	position.add(node.boundingBox.max).multiplyScalar(0.5);
		// 	position.applyMatrix4(octree.world);
		// 	let size = node.boundingBox.size();
		// 	let color = new Vector3(...SPECTRAL.get(node.level / 5));
		// 	renderer.drawBoundingBox(position, size, color);
		// }

		let numElements = node.geometry.numElements;
		passEncoder.draw(numElements, 1, 0, 0);
	}

};