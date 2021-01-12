
import {Vector3, Matrix4} from "../../math/math.js";

const vs = `
[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
	[[offset(128)]] screen_width : f32;
	[[offset(132)]] screen_height : f32;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> position : vec4<f32>;
[[location(1)]] var<in> color : vec4<f32>;

[[builtin(position)]] var<out> out_pos : vec4<f32>;
[[location(0)]] var<out> fragColor : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {
	
	var worldPos : vec4<f32> = position;
	var viewPos : vec4<f32> = uniforms.worldView * worldPos;
	out_pos = uniforms.proj * viewPos;

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

let vbo_lines = null;
let pipeline = null;
let uniformBuffer = null;
let uniformBindGroup = null;

function createBuffer(renderer, data){

	let {device} = renderer;

	let vbos = [];

	for(let entry of data.geometry.buffers){
		let {name, buffer} = entry;

		let vbo = device.createBuffer({
			size: buffer.byteLength,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
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
						format:  "uchar4norm",
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

function init(renderer){

	if(vbo_lines != null){
		return;
	}

	{ // create lines vbo

		let capacity = 10_000;

		let geometry = {
			buffers: [{
				name: "position",
				buffer: new Float32Array(2 * 3 * capacity),
			},{
				name: "color",
				buffer: new Uint8Array(2 * 4 * capacity),
			}]
		};
		let node = {geometry};

		vbo_lines = createBuffer(renderer, node);
	}

	{
		pipeline = createPipeline(renderer);

		let {device} = renderer;
		const uniformBufferSize = 2 * 4 * 16 + 8;

		uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		uniformBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [{
				binding: 0,
				resource: {buffer: uniformBuffer},
			}],
		});
	}

}

export function renderLines(renderer, pass, lines, camera){

	let {device} = renderer;

	init(renderer);

	{ // update uniforms

		{ // transform
			let world = new Matrix4();
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
	}

	let {passEncoder} = pass;

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, uniformBindGroup);

	{
		let position = new Float32Array(2 * 3 * lines.length);
		let color = new Uint8Array(2 * 4 * lines.length);

		for(let i = 0; i < lines.length; i++){
			let [start, end, c] = lines[i];

			position[6 * i + 0] = start.x;
			position[6 * i + 1] = start.y;
			position[6 * i + 2] = start.z;

			position[6 * i + 3] = end.x;
			position[6 * i + 4] = end.y;
			position[6 * i + 5] = end.z;

			color[8 * i + 0] = c.x;
			color[8 * i + 1] = c.y;
			color[8 * i + 2] = c.z;
			color[8 * i + 3] = 255;

			color[8 * i + 4] = c.x;
			color[8 * i + 5] = c.y;
			color[8 * i + 6] = c.z;
			color[8 * i + 7] = 255;
		}

		device.defaultQueue.writeBuffer(
			vbo_lines[0].vbo, 0,
			position.buffer, position.byteOffset, position.byteLength
		);

		device.defaultQueue.writeBuffer(
			vbo_lines[1].vbo, 0,
			color.buffer, color.byteOffset, color.byteLength
		);
	}

	passEncoder.setVertexBuffer(0, vbo_lines[0].vbo);
	passEncoder.setVertexBuffer(1, vbo_lines[1].vbo);

	passEncoder.draw(2 * lines.length, 1, 0, 0);
};