
import {Vector3, Matrix4, Geometry} from "potree";

const shaderSource = `
[[block]] struct Uniforms {
	view : mat4x4<f32>;
	proj : mat4x4<f32>;
	screen_width : f32;
	screen_height : f32;
	point_size : f32;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;

struct VertexIn{
	[[location(0)]] position : vec3<f32>;
	[[location(1)]] color : vec3<f32>;
	[[builtin(vertex_index)]] vertexID : u32;
};

struct VertexOut{
	[[builtin(position)]] position : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
};

struct FragmentIn{
	[[location(0)]] color : vec4<f32>;
};

struct FragmentOut{
	[[location(0)]] color : vec4<f32>;
};


[[stage(vertex)]]
fn main_vertex(vertex : VertexIn) -> VertexOut {

	var projPos = uniforms.proj * uniforms.view * vec4<f32>(vertex.position, 1.0);

	var vout : VertexOut;
	vout.position = projPos;
	vout.color = vec4<f32>(vertex.color, 1.0);

	return vout;
}

[[stage(fragment)]]
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	fout.color = fragment.color;

	return fout;
}
`;


export function render(meshes, drawstate){

	let {renderer} = drawstate;
	let {device} = renderer;
	let {passEncoder} = drawstate.pass;

	let pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: shaderSource}),
			entryPoint: "main_vertex",
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
						format: "unorm8x4",
					}],
				}
			]
		},
		fragment: {
			module: device.createShaderModule({code: shaderSource}),
			entryPoint: "main_fragment",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'back',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	let uniformBuffer = device.createBuffer({
		size: 256,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	let bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0,resource: {buffer: uniformBuffer}},
		],
	});

	{ // update uniforms

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
			view.setFloat32(136, 5.0, true);
		}

		renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
	}

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	for(let batch of meshes){
		let vboPosition = renderer.getGpuBuffer(batch.positions);
		let vboColor = renderer.getGpuBuffer(batch.colors);

		passEncoder.setVertexBuffer(0, vboPosition);
		passEncoder.setVertexBuffer(1, vboColor);

		let numVertices = batch.positions.length / 3;
		passEncoder.draw(numVertices, 1, 0, 0);
	}

};