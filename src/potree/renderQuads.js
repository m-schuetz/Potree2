
import { mat4, vec3 } from '../../libs/gl-matrix.js';

const vs = `
[[block]] struct Uniforms {
	[[offset(0)]] modelViewProjectionMatrix : mat4x4<f32>;
	[[offset(64)]] screen_width : f32;
	[[offset(68)]] screen_height : f32;

};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> pos_point : vec4<f32>;
[[location(1)]] var<in> pos_quad : vec4<f32>;
[[location(2)]] var<in> color : vec4<f32>;

[[builtin(position)]] var<out> out_pos : vec4<f32>;
[[location(0)]] var<out> fragColor : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {
	out_pos = uniforms.modelViewProjectionMatrix * pos_point;

	var fx : f32 = out_pos.x / out_pos.w;
	fx = fx + 2.0 * pos_quad.x / uniforms.screen_width;
	out_pos.x = fx * out_pos.w;

	var fy : f32 = out_pos.y / out_pos.w;
	fy = fy + 2.0 * pos_quad.y / uniforms.screen_height;
	out_pos.y = fy * out_pos.w;

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

let quad_position = new Float32Array([
	-1.0, -1.0, 0.0,
	 1.0, -1.0, 0.0,
	 1.0,  1.0, 0.0,
	-1.0, -1.0, 0.0,
	 1.0,  1.0, 0.0,
	-1.0,  1.0, 0.0,
]);
let vbo_quad = null;

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

function getVboQuad(renderer){
	if(!vbo_quad){

		let geometry = {
			buffers: [{
				name: "position",
				buffer: quad_position,
			}]
		};
		let node = {geometry};

		vbo_quad = createBuffer(renderer, node);
	}

	return vbo_quad;
};

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
		primitiveTopology: "triangle-list",
		depthStencilState: {
			depthWriteEnabled: true,
			depthCompare: "less",
			format: "depth24plus-stencil8",
		},
		vertexState: {
			vertexBuffers: [
				{ // point position
					arrayStride: 3 * 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float3",
					}],
				},{ // quad position
					arrayStride: 3 * 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "float3",
					}],
				},{ // color
					arrayStride: 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 2,
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

	let vbo_quad = getVboQuad(renderer);

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
		passEncoder.setVertexBuffer(1, vbo_quad[0].vbo);
		passEncoder.setVertexBuffer(2, nodeState.vbos[1].vbo);
	
		let numElements = node.geometry.numElements;
		// numElements = Math.min(numElements, 10);
		passEncoder.draw(6, numElements, 0, 0);
	}

};