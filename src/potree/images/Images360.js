
import {SceneNode, Vector3, Matrix4} from "potree";

let shaderCode = `

[[block]] struct Uniforms {
	worldView : mat4x4<f32>;
	proj : mat4x4<f32>;
	screen_width : f32;
	screen_height : f32;
	size: f32;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;

struct VertexIn{
	[[location(0)]] position : vec3<f32>;
	[[builtin(vertex_index)]] vertexID : u32;
};

struct VertexOut{
	[[builtin(position)]] position : vec4<f32>;
};

// struct FragmentIn{
// 	[[location(0)]] color : vec4<f32>;
// };

struct FragmentOut{
	[[location(0)]] color : vec4<f32>;
	[[location(1)]] point_id : u32;
};

[[stage(vertex)]]
fn main_vertex(vertex : VertexIn) -> VertexOut {

	var QUAD_POS : array<vec3<f32>, 6> = array<vec3<f32>, 6>(
		vec3<f32>(-1.0, -1.0, 0.0),
		vec3<f32>( 1.0, -1.0, 0.0),
		vec3<f32>( 1.0,  1.0, 0.0),

		vec3<f32>(-1.0, -1.0, 0.0),
		vec3<f32>( 1.0,  1.0, 0.0),
		vec3<f32>(-1.0,  1.0, 0.0),
	);

	var viewPos : vec4<f32> = uniforms.worldView * vec4<f32>(vertex.position, 1.0);
	var projPos : vec4<f32> = uniforms.proj * viewPos;

	let quadVertexIndex : u32 = vertex.vertexID % 6u;
	var pos_quad : vec3<f32> = QUAD_POS[quadVertexIndex];

	var fx : f32 = projPos.x / projPos.w;
	fx = fx + uniforms.size * pos_quad.x / uniforms.screen_width;
	projPos.x = fx * projPos.w;

	var fy : f32 = projPos.y / projPos.w;
	fy = fy + uniforms.size * pos_quad.y / uniforms.screen_height;
	projPos.y = fy * projPos.w;

	var vout : VertexOut;
	vout.position = projPos;

	return vout;
}

[[stage(fragment)]]
fn main_fragment() -> FragmentOut {

	var fout : FragmentOut;
	fout.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
	fout.point_id = 1u;

	return fout;
}

`;

let initialized = false;
let pipeline = null;
let uniformBuffer = null;
let bindGroup = null;

function init(renderer){

	if(initialized){
		return;
	}
	
	let {device} = renderer;

	let module = device.createShaderModule({code: shaderCode});

	pipeline = device.createRenderPipeline({
		vertex: {
			module,
			entryPoint: "main_vertex",
			buffers: [
				{ // position
					arrayStride: 3 * 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				}
			]
		},
		fragment: {
			module,
			entryPoint: "main_fragment",
			targets: [
				{format: "bgra8unorm"},
				{format: "r32uint"},
			],
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
		view.setFloat32(136, 10.0, true);
	}

	renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
}


export class Image360{

	constructor(){
		this.position = new Vector3();
		this.name = "";
	}

}

export class Images360 extends SceneNode{

	constructor(images){
		super(); 

		this.images = images;

		// test data
		let center = new Vector3(637227.1, 850869.3, 649.5);
		let n = 1_000;
		this.positions = new Float32Array(3 * n);
		for(let i = 0; i < n; i++){
			let u = i / n;
			let r = 2 * i;
			let x = center.x + r * Math.cos(4 * Math.PI * u);
			let y = center.y + r * Math.sin(4 * Math.PI * u);
			let z = center.z;

			let image = new Image360();
			image.position.set(x, y, z);
			image.name = `test_${i}`;

			this.images.push(image);
			
			this.positions[3 * i + 0] = x;
			this.positions[3 * i + 1] = y;
			this.positions[3 * i + 2] = z;
		}

		


	}

	render(drawstate){

		let {renderer} = drawstate;

		init(renderer);

		updateUniforms(drawstate);

		let {passEncoder} = drawstate.pass;

		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, bindGroup);

		let vboPosition = renderer.getGpuBuffer(this.positions);

		passEncoder.setVertexBuffer(0, vboPosition);

		let numVertices = this.positions.length / 3;
		// passEncoder.draw(numVertices, 1, 0, 0);
		passEncoder.draw(6, numVertices, 0, 0);

	}


}