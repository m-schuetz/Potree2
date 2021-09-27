
import {SceneNode, Matrix4} from "potree";

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
// - modify sampleFunction() down below to specify the shape of your parametric function
// - compilation is automatically done while typing
// - if you encounter compilation errors (red background), check the dev console for details

[[block]] struct Uniforms {
	worldView : mat4x4<f32>;
	proj : mat4x4<f32>;
	screen_width : f32;
	screen_height : f32;
	time : f32;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;

// see https://in.mathworks.com/help/symbolic/ezsurf.html
fn sampleFunction(u : f32, v : f32) -> vec4<f32> {

	var s = u * 2.0 * 3.1415;
	var t = v * 3.1415;

	var time = cos(2.5 * uniforms.time);
	time = pow(abs(time), 0.8) * sign(time);
	var r = 0.3 * (1.0 + sin(5.0 * s + 10.0 * t * time));
	var x = r * cos(s) * sin(t);
	var y = r * sin(s) * sin(t);
	var z = r * cos(t);

	return vec4<f32>(x, y, z, 1.0);
}

fn sampleNormal(u : f32, v : f32, spacing : f32) -> vec4<f32> {

	var epsilon = spacing * 0.1;

	var x_0 = sampleFunction(u - epsilon, v);
	var x_1 = sampleFunction(u + epsilon, v);
	var y_0 = sampleFunction(u, v - epsilon);
	var y_1 = sampleFunction(u, v + epsilon);

	var a = normalize(x_1 - x_0).xyz;
	var b = normalize(y_1 - y_0).xyz;
	var N = cross(a, b);

	return vec4<f32>(N, 1.0);
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

[[stage(vertex)]]
fn main_vertex_points(vertex : VertexIn) -> VertexOut {

	ignore(uniforms);

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
	var N = sampleNormal(u, v, spacing);
	N.b = 0.5;
	vout.color = N;

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
let pipeline = null;
let uniformBuffer = null;
let bindGroup = null;

function updatePipeline(renderer, code){
	let {device} = renderer;

	let shaderModule = device.createShaderModule({code: code});

	pipeline = device.createRenderPipeline({
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

	bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
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

					bindGroup = device.createBindGroup({
						layout: pipeline.getBindGroupLayout(0),
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
			}

			renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
		}

		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, bindGroup);

		passEncoder.draw(6 * 1_000_000, 1, 0, 0);


	}

}