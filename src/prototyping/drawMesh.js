
import { mat4, vec3 } from '../../libs/gl-matrix.js';

const vs = `
[[block]] struct Uniforms {
  [[offset(0)]] modelViewProjectionMatrix : mat4x4<f32>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> position : vec4<f32>;
[[location(1)]] var<in> color : vec4<f32>;

[[builtin(position)]] var<out> Position : vec4<f32>;
[[location(0)]] var<out> fragColor : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {
	Position = uniforms.modelViewProjectionMatrix * position;
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
				},{ // color
					arrayStride: 4 * 4,
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "float4",
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

function getState(renderer, node){

	let {device} = renderer;

	let state = states.get(node);

	if(!state){
		let vbos = createBuffer(renderer, node);
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

export function drawMesh(renderer, pass, node, camera){
	let {device, swapChain, depthTexture} = renderer;
	let size = renderer.getSize();

	let state = getState(renderer, node);

	let {uniformBuffer, pipeline, uniformBindGroup} = state;

	camera.aspect = size.width / size.height;
	camera.updateProj();

	let world = mat4.create();

	mat4.translate(world, world, vec3.fromValues(-0.5, -0.5, -0.5));

	let view = camera.view;
	let proj = camera.proj;

	let transformationMatrix = mat4.create();
	mat4.multiply(transformationMatrix, view, world);
	mat4.multiply(transformationMatrix, proj, transformationMatrix);

	device.defaultQueue.writeBuffer(
		uniformBuffer,
		0,
		transformationMatrix.buffer,
		transformationMatrix.byteOffset,
		transformationMatrix.byteLength
	);

	{
		let passEncoder = pass.commandEncoder.beginRenderPass(pass.renderPassDescriptor);
		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, uniformBindGroup);

		let vbos = state.vbos;
		for(let i = 0; i < vbos.length; i++){
			passEncoder.setVertexBuffer(i, vbos[i].vbo);
		}

		passEncoder.draw(node.vertexCount, 1, 0, 0);
		passEncoder.endPass();
	}
}