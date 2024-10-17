
import {SceneNode, Matrix4} from "potree";

const shaderSource = `
struct Uniforms {
	worldView : mat4x4<f32>,
	proj : mat4x4<f32>,
	screen_width : f32,
	screen_height : f32,
};

@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexIn{
	@builtin(vertex_index) index : u32,
};

struct VertexOut{
	@builtin(position) pos : vec4<f32>;
	@location(0) color : vec4<f32>,
};

struct FragmentIn{
	@location(0) color : vec4<f32>,
};

struct FragmentOut{
	@location(0) color : vec4<f32>,
};

@vertex
fn main_vertex_points(vertex : VertexIn) -> VertexOut {

	_ = uniforms;

	var u = f32(vertex.index) / 1000000.0;
	var z = 2.0 * u - 1.0;
	var r_xy = sqrt(1.0 - z * z);
	var x = r_xy * sin(500.0 * u);
	var y = r_xy * cos(500.0 * u);

	var worldPos = vec4<f32>(x, y, z, 1.0);
	var viewPos = uniforms.worldView * worldPos;
	var projPos = uniforms.proj * viewPos;

	var vout : VertexOut;
	vout.pos = projPos;
	vout.color = vec4<f32>(u, 0.0, 0.0, 1.0);

	return vout;
}

// fn sampleFunction(u : f32, v : f32) -> vec4<f32> {

// 	var x = 2.0 * u - 1.0;
// 	var y = 2.0 * v - 1.0;
// 	var dd = x * x + y * y;
// 	var z = (1.5 - dd) * 0.2 * cos(15.0 * dd);

// 	return vec4<f32>(x, y, z, 1.0);
// }

// https://in.mathworks.com/help/symbolic/ezsurf.html
fn sampleFunction(u : f32, v : f32) -> vec4<f32> {

	var s = u * 2.0 * 3.1415;
	var t = v * 3.1415;

	var r = 0.2 * (2.0 + sin(7.0 * s + 5.0 * t));
	var x = r * cos(s) * sin(t);
	var y = r * sin(s) * sin(t);
	var z = r * cos(t);

	return vec4<f32>(x, y, z, 1.0);
}

fn sampleNormal(u : f32, v : f32, spacing : f32) -> vec4<f32> {

	var x_0 = sampleFunction(u - spacing, v);
	var x_1 = sampleFunction(u + spacing, v);
	var y_0 = sampleFunction(u, v - spacing);
	var y_1 = sampleFunction(u, v + spacing);

	var a = normalize(x_1 - x_0).xyz;
	var b = normalize(y_1 - y_0).xyz;
	var N = cross(a, b);

	return vec4<f32>(N, 1.0);
}

@vertex
fn main_vertex_triangles(vertex : VertexIn) -> VertexOut {

	_ = uniforms;

	var QUAD_OFFSETS : array<vec3<f32>, 6> = array<vec3<f32>, 6>(
		vec3<f32>( 0.0, 0.0, 0.0),
		vec3<f32>( 1.0, 0.0, 0.0),
		vec3<f32>( 1.0, 1.0, 0.0),

		vec3<f32>( 0.0, 0.0, 0.0),
		vec3<f32>( 1.0, 1.0, 0.0),
		vec3<f32>( 0.0, 1.0, 0.0),
	);

	var spacing = 1.0 / 1000.0;

	var localVertexIndex = vertex.index % 6u;
	var ix = (vertex.index / 6u) % 1000u;
	var iy = (vertex.index / 6u) / 1000u;

	var u = f32(ix) / 1000.0;
	var v = f32(iy) / 1000.0;

	var worldPos = sampleFunction(
		u + QUAD_OFFSETS[localVertexIndex].x * spacing,
		v + QUAD_OFFSETS[localVertexIndex].y * spacing
	);

	var viewPos = uniforms.worldView * worldPos;
	var projPos = uniforms.proj * viewPos;

	var vout : VertexOut;
	vout.pos = projPos;
	// vout.color = vec4<f32>(u, v, 0.0, 1.0);
	vout.color = sampleNormal(u, v, spacing);

	return vout;
}

@fragment
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	_ = uniforms;

	var fout : FragmentOut;
	fout.color = fragment.color;

	return fout;
}
`;

let initialized = null;
let pipeline = null;
let uniformBuffer = null;
let bindGroup = null;

function initialize(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;
	
	let shaderModule = device.createShaderModule({code: shaderSource});

	pipeline = device.createRenderPipeline({
		label: "parametricFunction",
		layout: "auto",
		vertex: {
			module: shaderModule,
			entryPoint: "main_vertex_triangles",
			buffers: []
		},
		fragment: {
			module: shaderModule,
			entryPoint: "main_fragment",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	let uniformBufferSize = 256;

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

	initialized = true;
}

export class ParametricFunction extends SceneNode {

	constructor(name){
		super(name);
	}

	render(drawstate){

		let {renderer} = drawstate;
		
		initialize(renderer);

		let {passEncoder} = drawstate.pass;

		{ // update uniforms
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

		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, bindGroup);

		passEncoder.draw(6 * 1_000_000, 1, 0, 0);


	}

}