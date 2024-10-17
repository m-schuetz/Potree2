
import {Geometry, Vector3, Matrix4} from "potree";

const vs = `
struct Uniforms {
	worldView : mat4x4<f32>,
	proj : mat4x4<f32>,
	screen_width : f32,
	screen_height : f32,
};

@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexIn{
	@location(0)         position  : vec4<f32>,
	@location(1)         color     : vec4<f32>,
};

struct VertexOut{
	@builtin(position)   position  : vec4<f32>,
	@location(0)         color     : vec4<f32>,
};


@vertex
fn main(vertex : VertexIn) -> VertexOut {

	var vout : VertexOut;
	
	var worldPos : vec4<f32> = vertex.position;
	var viewPos : vec4<f32> = uniforms.worldView * worldPos;
	
	vout.position = uniforms.proj * viewPos;
	vout.color = vertex.color;

	return vout;
}
`;

const fs = `

struct FragmentIn{
	@builtin(position) position  : vec4<f32>;
	@location(0) color : vec4<f32>,
};

struct FragmentOut{
	@location(0) color : vec4<f32>,
	@builtin(frag_depth) depth : f32,
};

@fragment
fn main(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	fout.color = fragment.color;

	fout.depth = fragment.position.z * 1.001;

	return fout;
}
`;


let vbo_lines = null;
let initialized = false;
let pipeline = null;
let uniformBuffer = null;
let bindGroup = null;
let capacity = 100_000;

function createPipeline(renderer){

	let {device} = renderer;

	pipeline = device.createRenderPipeline({
		label: "renderLines",
		layout: "auto",
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
						format:  "unorm8x4",
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
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	return pipeline;
}

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

function init(renderer){

	if(initialized){
		return;
	}

	{ // create lines vbo

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

		view.setUint32(128, size.width, true);
		view.setUint32(132, size.height, true);
	}

	renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
}

export function render(lines, drawstate){

	let {renderer} = drawstate;
	let {device} = renderer;

	init(renderer);

	updateUniforms(drawstate);

	let {passEncoder} = drawstate.pass;

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

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

		device.queue.writeBuffer(
			vbo_lines[0].vbo, 0,
			position.buffer, position.byteOffset, position.byteLength
		);

		device.queue.writeBuffer(
			vbo_lines[1].vbo, 0,
			color.buffer, color.byteOffset, color.byteLength
		);
	}

	passEncoder.setVertexBuffer(0, vbo_lines[0].vbo);
	passEncoder.setVertexBuffer(1, vbo_lines[1].vbo);

	passEncoder.draw(2 * lines.length, 1, 0, 0);

};