
import { mat4, vec3 } from '../../../libs/gl-matrix.js';

const vs = `
[[block]] struct Uniforms {
  [[offset(0)]] modelViewProjectionMatrix : mat4x4<f32>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> position : vec4<f32>;

[[builtin(position)]] var<out> Position : vec4<f32>;
[[location(0)]] var<out> fragColor : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {
	Position = uniforms.modelViewProjectionMatrix * position;
	fragColor = vec4<f32>(1.0, 0.0, 0.0, 1.0);
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

let states = new Map();

function createBuffer(renderer, data){

	let {device} = renderer;

	let vbos = [];

	for(let entry of data.buffers){
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

function createPipeline(renderer, vbos){

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
		primitiveTopology: "line-list",
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
				}
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

function getState(renderer, node){

	let {device} = renderer;

	let state = states.get(node);

	if(!state){
		let vbos = createBuffer(renderer, node.geometry);
		let pipeline = createPipeline(renderer);

		const uniformBufferSize = 4 * 16; 

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
			vbos: vbos,
			pipeline: pipeline,
			uniformBuffer: uniformBuffer,
			uniformBindGroup: uniformBindGroup,
		};

		states.set(node, state);

	}

	return state;
}

export function render(renderer, pass, node, camera){
	let {device} = renderer;

	let state = getState(renderer, node);

	{ // update uniforms
		let glWorld = mat4.create();
		mat4.set(glWorld, ...node.world.elements);

		let view = camera.view;
		let proj = camera.proj;

		let transform = mat4.create();
		mat4.multiply(transform, view, glWorld);
		mat4.multiply(transform, proj, transform);

		let {uniformBuffer} = state;
		device.defaultQueue.writeBuffer(
			uniformBuffer, 0,
			transform.buffer, transform.byteOffset, transform.byteLength
		);
	}

	{ // render
		let {passEncoder} = pass;
		let {pipeline, uniformBindGroup} = state;

		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, uniformBindGroup);

		let vbos = state.vbos;
		passEncoder.setVertexBuffer(0, vbos[0].vbo);

		passEncoder.draw(node.geometry.numElements, 1, 0, 0);
		
	}
}