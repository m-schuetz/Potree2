
import {Potree, SceneNode, Matrix4} from "potree";

//====================================================
//====================================================
// fn sampleFunction(u : f32, v : f32) -> vec4<f32> {

// 	var s = 2.0 * 3.14159 * u;
// 	var t = 2.0 * 3.14159 * v;

// 	var a = 1.0;
// 	var b = 0.3;

// 	var k = (4.0 * t) % (PI);
// 	var r = 1.0 + 0.3 * cos(2.0 * k);

// 	var x = r * (a + b * cos(s)) * cos(t);
// 	var y = r * (a + b * cos(s)) * sin(t);
// 	var z = b * sin(s) + 0.2 * sin(2.0 * k);

// 	return vec4<f32>(x, y, z, 1.0);
// }
//====================================================
//====================================================
//====================================================
//====================================================
//====================================================
//====================================================
//====================================================





const shaderSource = `
// - modify samplePosition() to specify the shape of your parametric function
// - compilation is automatically done while typing
// - if you encounter compilation errors (red background), 
//   check the dev console for details

[[block]] struct Uniforms {
	worldView          : mat4x4<f32>;
	proj               : mat4x4<f32>;
	screen_width       : f32;
	screen_height      : f32;
	time               : f32;
	resolution         : u32;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;

// see https://in.mathworks.com/help/symbolic/ezsurf.html
fn samplePosition(u : f32, v : f32) -> vec4<f32> {

	var s = u * 2.0 * 3.1415;
	var t = v * 3.1415;

	var time = cos(uniforms.time);
	time = pow(abs(time), 0.8) * sign(time);
	var r = 0.3 * (1.0 + sin(5.0 * s + 10.0 * t * time));
	var x = r * cos(s) * sin(t);
	var y = r * sin(s) * sin(t);
	var z = r * cos(t);

	return vec4<f32>(x, y, z, 1.0);
}

fn sampleColor(u : f32, v : f32, spacing : f32) -> vec4<f32> {

	var delta = 40.0 / 1000.0;

	var x_0 = samplePosition(u - delta, v);
	var x_1 = samplePosition(u + delta, v);
	var y_0 = samplePosition(u, v - delta);
	var y_1 = samplePosition(u, v + delta);

	var a = normalize(x_1 - x_0).xyz;
	var b = normalize(y_1 - y_0).xyz;
	var N = cross(a, b);

	var color = vec4<f32>(
		pow(1.0 * abs(N.x), 0.7),
		pow(1.0 * abs(N.y), 1.1),
		pow(abs(N.z), 0.2),
		1.0,
	);

	return color;
}

struct VertexIn{
	[[builtin(vertex_index)]] index : u32;
};

struct VertexOut{
	[[builtin(position)]] pos : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
};

struct FragmentIn{
	[[location(0)]] color : vec4<f32>;
};

struct FragmentOut{
	[[location(0)]] color : vec4<f32>;
};

// vertex shader for triangle primitives
[[stage(vertex)]]
fn main_vertex_triangles(vertex : VertexIn) -> VertexOut {

	ignore(uniforms);

	var QUAD_OFFSETS : array<vec3<f32>, 6> = array<vec3<f32>, 6>(
		vec3<f32>( 0.0, 0.0, 0.0),
		vec3<f32>( 1.0, 0.0, 0.0),
		vec3<f32>( 1.0, 1.0, 0.0),

		vec3<f32>( 0.0, 0.0, 0.0),
		vec3<f32>( 1.0, 1.0, 0.0),
		vec3<f32>( 0.0, 1.0, 0.0),
	);
	
	var spacing = 1.0 / f32(uniforms.resolution);

	var localVertexIndex = vertex.index % 6u;
	var ix = (vertex.index / 6u) % uniforms.resolution;
	var iy = (vertex.index / 6u) / uniforms.resolution;

	var u = f32(ix) / f32(uniforms.resolution);
	var v = f32(iy) / f32(uniforms.resolution);

	var worldPos = samplePosition(
		u + QUAD_OFFSETS[localVertexIndex].x * spacing,
		v + QUAD_OFFSETS[localVertexIndex].y * spacing
	);

	var viewPos = uniforms.worldView * worldPos;
	var projPos = uniforms.proj * viewPos;

	var vout : VertexOut;
	vout.pos = projPos;
	vout.color = sampleColor(u, v, spacing);

	return vout;
}

// vertex shader for point primitives
[[stage(vertex)]]
fn main_vertex_points(vertex : VertexIn) -> VertexOut {

	ignore(uniforms);
	
	var spacing = 1.0 / f32(uniforms.resolution);

	var localVertexIndex = vertex.index % 6u;
	var ix = vertex.index % uniforms.resolution;
	var iy = vertex.index / uniforms.resolution;

	var u = f32(ix) / f32(uniforms.resolution);
	var v = f32(iy) / f32(uniforms.resolution);

	var worldPos = samplePosition(u, v);
	var viewPos = uniforms.worldView * worldPos;
	var projPos = uniforms.proj * viewPos;

	var vout : VertexOut;
	vout.pos = projPos;
	vout.color = sampleColor(u, v, spacing);

	return vout;
}

[[stage(fragment)]]
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	ignore(uniforms);

	var fout : FragmentOut;
	fout.color = fragment.color;

	return fout;
}



`;




let initialized = null;
let pipeline_triangles = null;
let pipeline_points = null;

let uniformBuffer = null;

let bindGroup_triangles = null;
let bindGroup_points = null;

function updatePipeline(renderer, code){
	let {device} = renderer;

	let shaderModule = device.createShaderModule({code: code});

	pipeline_triangles = device.createRenderPipeline({
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

	pipeline_points = device.createRenderPipeline({
		vertex: {
			module: shaderModule,
			entryPoint: "main_vertex_points",
			buffers: []
		},
		fragment: {
			module: shaderModule,
			entryPoint: "main_fragment",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'point-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});
}

function initialize(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	updatePipeline(renderer, shaderSource);

	let uniformBufferSize = 256;

	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	bindGroup_triangles = device.createBindGroup({
		layout: pipeline_triangles.getBindGroupLayout(0),
		entries: [
			{binding: 0,resource: {buffer: uniformBuffer}},
		],
	});
	bindGroup_points = device.createBindGroup({
		layout: pipeline_points.getBindGroupLayout(0),
		entries: [
			{binding: 0,resource: {buffer: uniformBuffer}},
		],
	});
	
	{
		let elCode = document.getElementById("code");

		elCode.value = shaderSource;

		elCode.addEventListener('input', async function(e) {
			// if(e.ctrlKey && e.key === "Enter")
			{

				
				let shaderModule = device.createShaderModule({code: elCode.value});

				let info = await shaderModule.compilationInfo();

				let isError = info.messages.find(message => message.type === "error") != null;

				if(isError){
					console.error("ERROR");

					elCode.style.backgroundColor = "#fdd";
				}else{
					console.info("compiled");

					updatePipeline(renderer, elCode.value);

					bindGroup_triangles = device.createBindGroup({
						layout: pipeline_triangles.getBindGroupLayout(0),
						entries: [
							{binding: 0,resource: {buffer: uniformBuffer}},
						],
					});
					bindGroup_points = device.createBindGroup({
						layout: pipeline_points.getBindGroupLayout(0),
						entries: [
							{binding: 0,resource: {buffer: uniformBuffer}},
						],
					});

					elCode.style.backgroundColor = "#dfd";
				}


			}

			console.log(e.key);
		});
	}

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

		let primitive = Potree.settings.primitive;

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
				view.setFloat32(136, performance.now() / 1000, true);
				view.setUint32(140, Potree.settings.resolution, true);
			}

			renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
		}

		if(primitive === "triangles"){
			passEncoder.setPipeline(pipeline_triangles);
			passEncoder.setBindGroup(0, bindGroup_triangles);

			let numPoints = (1 + Potree.settings.resolution) ** 2;
		
			passEncoder.draw(6 * numPoints, 1, 0, 0);
		}else if(primitive === "points"){
			passEncoder.setPipeline(pipeline_points);
			passEncoder.setBindGroup(0, bindGroup_points);

			let numPoints = (1 + Potree.settings.resolution) ** 2;
		
			passEncoder.draw(numPoints, 1, 0, 0);
		}

	}

}