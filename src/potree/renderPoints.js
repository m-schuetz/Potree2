
import { mat4, vec3 } from '../../libs/gl-matrix.js';
import { Vector3 } from '../math/Vector3.js';


const vs = `
[[block]] struct Uniforms {
	[[offset(0)]] modelViewProjectionMatrix : mat4x4<f32>;
	[[offset(64)]] screen_width : f32;
	[[offset(68)]] screen_height : f32;

};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> pos_point : vec4<f32>;
[[location(1)]] var<in> color : vec4<f32>;

[[builtin(position)]] var<out> out_pos : vec4<f32>;
[[location(0)]] var<out> fragColor : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {
	out_pos = uniforms.modelViewProjectionMatrix * pos_point;
	# out_pos = uniforms.modelViewProjectionMatrix * (pos_point * vec4<f32>(0.5, 0.5, 0.5, 1.0));

	fragColor = color;

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

function createBuffer(renderer, data){

	let {device} = renderer;

	let vbos = [];

	for(let entry of data.geometry.buffers){
		let {name, buffer} = entry;

		let vbo = device.createBuffer({
			size: buffer.byteLength,
			usage: GPUBufferUsage.VERTEX,
			mappedAtCreation: true,
		});

		let type = buffer.constructor;
		new type(vbo.getMappedRange()).set(buffer);
		vbo.unmap();

		vbos.push({
			name: name,
			vbo: vbo,
		});
	}

	return vbos;
}

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

		const uniformBufferSize = 4 * 16 + 8;

		const uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const uniformBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [{
				binding: 0,
				resource: {buffer: uniformBuffer},
			}],
		});

		state = {
			pipeline: pipeline,
			uniformBuffer: uniformBuffer,
			uniformBindGroup: uniformBindGroup,
		};

		octreeStates.set(node, state);

	}

	return state;
}

function getNodeState(renderer, node){
	let state = nodeStates.get(node);

	if(!state){
		let vbos = createBuffer(renderer, node);

		state = {vbos};
		nodeStates.set(node, state);
	}

	return state;
}

export function render(renderer, pass, octree, camera){

	let {device} = renderer;

	let octreeState = getOctreeState(renderer, octree);

	{ // update uniforms
		let {uniformBuffer} = octreeState;

		{ // transform
			let glWorld = mat4.create();
			mat4.set(glWorld, ...octree.world.elements);

			let view = camera.view;
			let proj = camera.proj;

			let flip = mat4.create();
			mat4.set(flip,
				1, 0, 0, 0,
				0, 0, -1, 0,
				0, 1, 0, 0,
				0, 0, 0, 1,
			);
			let transform = mat4.create();
			mat4.multiply(transform, flip, glWorld);
			mat4.multiply(transform, view, transform);
			mat4.multiply(transform, proj, transform);

			device.defaultQueue.writeBuffer(
				uniformBuffer, 0,
				transform.buffer, transform.byteOffset, transform.byteLength
			);
		}

		{ // screen size
			let size = renderer.getSize();
			let data = new Float32Array([size.width, size.height]);
			device.defaultQueue.writeBuffer(
				uniformBuffer,
				4 * 16,
				data.buffer,
				data.byteOffset,
				data.byteLength
			);
		}
	}

	let {passEncoder} = pass;
	let {pipeline, uniformBindGroup} = octreeState;

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, uniformBindGroup);

	let nodes = octree.visibleNodes;

	for(let node of nodes){
		let nodeState = getNodeState(renderer, node);

		passEncoder.setVertexBuffer(0, nodeState.vbos[0].vbo);
		passEncoder.setVertexBuffer(1, nodeState.vbos[1].vbo);

		if(octree.showBoundingBox === true){
			let position = node.boundingBox.min.clone();
			position.add(node.boundingBox.max).multiplyScalar(0.5);
			position.applyMatrix4(octree.world);
			let size = node.boundingBox.size();
			let color = new Vector3(1, 1, 1);
			renderer.drawBoundingBox(position, size, color);
		}
	
		let numElements = node.geometry.numElements;
		// numElements = Math.min(numElements, 10);
		passEncoder.draw(numElements, 1, 0, 0);
	}

};