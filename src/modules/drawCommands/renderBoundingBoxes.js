
import { mat4, vec3 } from '../../../libs/gl-matrix.js';

const vs = `
[[block]] struct Uniforms {
	[[offset(0)]] viewProj : mat4x4<f32>;
	[[offset(64)]] screen_width : f32;
	[[offset(68)]] screen_height : f32;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> box_pos : vec4<f32>;
[[location(1)]] var<in> box_scale : vec4<f32>;
[[location(2)]] var<in> point_pos : vec4<f32>;
[[location(3)]] var<in> box_color : vec4<f32>;

[[builtin(position)]] var<out> out_pos : vec4<f32>;
[[location(0)]] var<out> fragColor : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {
	# out_pos = uniforms.viewProj * (box_pos + point_pos * box_scale);

	var worldPos : vec4<f32> = box_pos + point_pos * box_scale;
	worldPos.w = 1.0;

	out_pos = uniforms.viewProj * worldPos;

	fragColor = box_color;

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

let box_position = new Float32Array([
	-0.5, -0.5, -0.5, // 0
	-0.5, -0.5,  0.5, // 1
	-0.5,  0.5, -0.5, // 2
	-0.5,  0.5,  0.5, // 3
	 0.5, -0.5, -0.5, // 4
	 0.5, -0.5,  0.5, // 5
	 0.5,  0.5, -0.5, // 6
	 0.5,  0.5,  0.5, // 7
]);

let box_elements = new Uint32Array([
	// bottom
	0, 2,
	2, 6,
	4, 6,
	4, 0,

	// top
	1, 3,
	3, 7,
	5, 7,
	5, 1,
	
	// bottom to top
	0, 1,
	2, 3, 
	4, 5,
	6, 7,
]);

let vbo_box = null;
let vbo_boxes = null;
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
				{ // box position
					arrayStride: 3 * 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float3",
					}],
				},{ // box scale
					arrayStride: 3 * 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "float3",
					}],
				},{ // box-vertices position
					arrayStride: 3 * 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 2,
						offset: 0,
						format: "float3",
					}],
				},{ // box color
					arrayStride: 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 3,
						offset: 0,
						format: "uchar4norm",
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

	if(vbo_box != null){
		return;
	}

	{ // create box vbo
		let geometry = {
			buffers: [{
				name: "position",
				buffer: box_position,
			},{
				name: "elements",
				buffer: box_elements,
			}]
		};
		let node = {geometry};

		vbo_box = createBuffer(renderer, node);
	}

	{ // create boxes vbo

		let capacity = 10_000;

		let geometry = {
			buffers: [{
				name: "position",
				buffer: new Float32Array(3 * capacity),
			},{
				name: "scale",
				buffer: new Float32Array(3 * capacity),
			},{
				name: "color",
				buffer: new Uint8Array(4 * capacity),
			}]
		};
		let node = {geometry};

		vbo_boxes = createBuffer(renderer, node);
	}

	{
		pipeline = createPipeline(renderer);

		let {device} = renderer;
		const uniformBufferSize = 4 * 16 + 8;

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

export function renderBoundingBoxes(renderer, pass, boxes, camera){

	let {device} = renderer;

	init(renderer);

	{ // update uniforms

		{ // transform
			let view = camera.view;
			let proj = camera.proj;

			// let flip = mat4.create();
			// mat4.set(flip,
			// 	1, 0, 0, 0,
			// 	0, 0, -1, 0,
			// 	0, 1, 0, 0,
			// 	0, 0, 0, 1,
			// );

			let transform = mat4.create();
			// mat4.multiply(transform, flip, transform);
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

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, uniformBindGroup);


	{
		let position = new Float32Array(3 * boxes.length);
		let scale = new Float32Array(3 * boxes.length);
		let color = new Uint8Array(4 * boxes.length);

		for(let i = 0; i < boxes.length; i++){
			let box = boxes[i];
			let pos = box[0];

			position[3 * i + 0] = pos.x;
			position[3 * i + 1] = pos.y;
			position[3 * i + 2] = pos.z;

			scale[3 * i + 0] = box[1].x;
			scale[3 * i + 1] = box[1].y;
			scale[3 * i + 2] = box[1].z;

			color[4 * i + 0] = box[2].x;
			color[4 * i + 1] = box[2].y;
			color[4 * i + 2] = box[2].z;
			color[4 * i + 3] = 255;
		}

		device.defaultQueue.writeBuffer(
			vbo_boxes[0].vbo, 0,
			position.buffer, position.byteOffset, position.byteLength
		);

		device.defaultQueue.writeBuffer(
			vbo_boxes[1].vbo, 0,
			scale.buffer, scale.byteOffset, scale.byteLength
		);

		device.defaultQueue.writeBuffer(
			vbo_boxes[2].vbo, 0,
			color.buffer, color.byteOffset, color.byteLength
		);
	}

	passEncoder.setVertexBuffer(0, vbo_boxes[0].vbo);
	passEncoder.setVertexBuffer(1, vbo_boxes[1].vbo);
	passEncoder.setVertexBuffer(2, vbo_box[0].vbo);
	passEncoder.setVertexBuffer(3, vbo_boxes[2].vbo);
	passEncoder.setIndexBuffer(vbo_box[1].vbo, "uint32");

	let numBoxes = boxes.length;
	let numVertices = box_elements.length;
	passEncoder.drawIndexed(numVertices, numBoxes, 0, 0);


};