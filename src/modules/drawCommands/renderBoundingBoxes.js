
import {Vector3, Matrix4, Geometry} from "potree";
import {cube, cube_wireframe} from "../geometries/cube.js";


const vs = `
[[block]] struct Uniforms {
	worldView : mat4x4<f32>;
	proj : mat4x4<f32>;
	screen_width : f32;
	screen_height : f32;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;

struct VertexIn{
	[[location(0)]] box_pos : vec3<f32>;
	[[location(1)]] box_scale : vec3<f32>;
	[[location(2)]] point_pos : vec3<f32>;
	[[location(3)]] box_color : vec4<f32>;
	[[location(4)]] in_dir : vec3<f32>;
	[[builtin(vertex_index)]] vertexID : u32;
};

struct VertexOut{
	[[builtin(position)]] pos : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
};



[[stage(vertex)]]
fn main(vertex : VertexIn) -> VertexOut {

	var worldPos : vec4<f32> = vec4<f32>(vertex.box_pos + vertex.point_pos * vertex.box_scale, 1.0);
	worldPos.w = 1.0;
	var viewPos : vec4<f32> = uniforms.worldView * worldPos;

	var vout : VertexOut;
	vout.pos = uniforms.proj * viewPos;
	vout.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);

	return vout;
}
`;

const fs = `

struct FragmentIn{
	[[location(0)]] color : vec4<f32>;
};

struct FragmentOut{
	[[location(0)]] color : vec4<f32>;
};

[[stage(fragment)]]
fn main(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	fout.color = fragment.color;

	return fout;
}
`;

let initialized = false;
let pipeline = null;
let geometry_boxes = null;
let uniformBuffer = null;
let bindGroup = null;
let capacity = 10_000;

function createPipeline(renderer){

	let {device} = renderer;
	
	pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
			buffers: [
				{ // box position
					arrayStride: 3 * 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				},{ // box scale
					arrayStride: 3 * 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "float32x3",
					}],
				},{ // box-vertices position
					arrayStride: 3 * 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 2,
						offset: 0,
						format: "float32x3",
					}],
				},{ // box color
					arrayStride: 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 3,
						offset: 0,
						format: "unorm8x4",
					}],
				},{ // box line directions
					arrayStride: 3 * 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 4,
						offset: 0,
						format: "float32x3",
					}],
				}
			]
		},
		fragment: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'line-list',
			cullMode: 'back',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	return pipeline;
}

function init(renderer){

	if(initialized){
		return;
	}

	geometry_boxes = new Geometry({
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
	});

	{
		pipeline = createPipeline(renderer);

		let {device} = renderer;
		const uniformBufferSize = 256;

		uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0,resource: {buffer: uniformBuffer}},
			],
		});
	}

	initialized = true;

}

function updateUniforms(drawstate){

	let {renderer, camera} = drawstate;

	let data = new ArrayBuffer(256);
	let f32 = new Float32Array(data);
	let view = new DataView(data);

	{ // transform
		let world = new Matrix4();
		let view = camera.view;
		let worldView = new Matrix4().multiplyMatrices(view, world);

		f32.set(worldView.elements, 0);
		f32.set(camera.proj.elements, 16);
	}

	{ // misc
		let size = renderer.getSize();

		view.setFloat32(128, size.width, true);
		view.setFloat32(132, size.height, true);
	}

	renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
}

export function render(boxes, drawstate){

	let {renderer} = drawstate;
	let {device} = renderer;

	init(renderer);

	updateUniforms(drawstate);

	let {passEncoder} = drawstate.pass;

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	let position = geometry_boxes.buffers.find(g => g.name === "position").buffer;
	let scale = geometry_boxes.buffers.find(g => g.name === "scale").buffer;
	let color = geometry_boxes.buffers.find(g => g.name === "color").buffer;
	let vboPosition = renderer.getGpuBuffer(position);
	let vboScale = renderer.getGpuBuffer(scale);
	let vboColor = renderer.getGpuBuffer(color);
	{
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

		device.queue.writeBuffer(vboPosition, 0, position.buffer, 0, position.byteLength);
		device.queue.writeBuffer(vboScale, 0, scale.buffer, 0, scale.byteLength);
		device.queue.writeBuffer(vboColor, 0, color.buffer, 0, color.byteLength);
	}

	{ // wireframe
		let boxVertices = cube_wireframe.buffers.find(b => b.name === "position").buffer;
		let boxLineDirections = cube_wireframe.buffers.find(b => b.name === "direction").buffer;
		let vboBoxVertices = renderer.getGpuBuffer(boxVertices);
		let vboBoxLineDirections = renderer.getGpuBuffer(boxLineDirections);

		passEncoder.setVertexBuffer(0, vboPosition);
		passEncoder.setVertexBuffer(1, vboScale);
		passEncoder.setVertexBuffer(2, vboBoxVertices);
		passEncoder.setVertexBuffer(3, vboColor);
		passEncoder.setVertexBuffer(4, vboBoxLineDirections);
		// passEncoder.setIndexBuffer(vboBoxIndices, "uint32");

		let numBoxes = boxes.length;
		let numVertices = cube_wireframe.numElements;
		passEncoder.draw(numVertices, numBoxes, 0, 0);
	}

	// { // solid
	// 	let boxVertices = cube.buffers.find(b => b.name === "position").buffer;
	// 	let vboBoxVertices = renderer.getGpuBuffer(boxVertices);

	// 	passEncoder.setVertexBuffer(0, vboPosition);
	// 	passEncoder.setVertexBuffer(1, vboScale);
	// 	passEncoder.setVertexBuffer(2, vboBoxVertices);
	// 	passEncoder.setVertexBuffer(3, vboColor);
	// 	// passEncoder.setIndexBuffer(vboBoxIndices, "uint32");

	// 	let numBoxes = boxes.length;
	// 	let numVertices = cube.numElements;
	// 	passEncoder.draw(numVertices, numBoxes, 0, 0);
	// }


};