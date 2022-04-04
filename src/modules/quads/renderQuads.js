
import {Vector3, Matrix4} from "potree";

const vs = `
struct Uniforms {
	worldView      : mat4x4<f32>,
	proj           : mat4x4<f32>,
	screen_width   : f32,
	screen_height  : f32,
	point_size     : f32,
	size           : f32,
	min_pixel_size : f32,
	max_pixel_size : f32,
};

@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexIn{
	@location(0) position : vec3<f32>,
	@builtin(vertex_index) vertexID : u32,
};

struct VertexOut{
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec4<f32>,
	@location(1) uv    : vec2<f32>,
	@location(2) viewPos : vec3<f32>,
};


@stage(vertex)
fn main(vertex : VertexIn) -> VertexOut {

	var QUAD_POS : array<vec3<f32>, 6> = array<vec3<f32>, 6>(
		vec3<f32>(-1.0, -1.0, 0.0),
		vec3<f32>( 1.0, -1.0, 0.0),
		vec3<f32>( 1.0,  1.0, 0.0),

		vec3<f32>(-1.0, -1.0, 0.0),
		vec3<f32>( 1.0,  1.0, 0.0),
		vec3<f32>(-1.0,  1.0, 0.0),
	);

	var viewPos : vec4<f32> = uniforms.worldView * vec4<f32>(vertex.position, 1.0);
	var viewTopPos = viewPos + vec4<f32>(0.0, uniforms.size, 0.0, 0.0);
	var projPos : vec4<f32> = uniforms.proj * viewPos;
	var projTopPos : vec4<f32> = uniforms.proj * viewTopPos;

	var screenY = uniforms.screen_height * (0.5 + 0.5 * projPos.y / projPos.w);
	var screenTopY = uniforms.screen_height * (0.5 + 0.5 * projTopPos.y / projTopPos.w);
	var radius = max(abs(screenTopY - screenY), 0.0) / 2.0;

	radius = min(radius, uniforms.max_pixel_size);
	radius = max(radius, uniforms.min_pixel_size);

	let quadVertexIndex : u32 = vertex.vertexID % 6u;
	var pos_quad : vec3<f32> = QUAD_POS[quadVertexIndex];

	var fx : f32 = projPos.x / projPos.w;
	fx = fx + radius * pos_quad.x / uniforms.screen_width;
	projPos.x = fx * projPos.w;

	var fy : f32 = projPos.y / projPos.w;
	fy = fy + radius * pos_quad.y / uniforms.screen_height;
	projPos.y = fy * projPos.w;

	var vout : VertexOut;
	vout.position = projPos;
	vout.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
	vout.uv = vec2<f32>(
		pos_quad.x * 0.5 + 0.5, 
		pos_quad.y * 0.5 + 0.5,
	);
	// vout.depth = projPos.z / projPos.w;
	vout.viewPos = viewPos.xyz;

	return vout;
}
`;

const fs = `

struct FragmentIn{
	@location(0) color : vec4<f32>,
	@location(1) uv    : vec2<f32>,
	@location(2) viewPos : vec3<f32>,
};

struct FragmentOut{
	@location(0) color : vec4<f32>,
	@builtin(frag_depth) depth : f32,
};

struct Uniforms {
	worldView : mat4x4<f32>,
	proj : mat4x4<f32>,
	screen_width : f32,
	screen_height : f32,
	point_size : f32,
	size : f32,
	min_pixel_size : f32,
	max_pixel_size: f32,
};

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(2) @group(0) var mySampler: sampler;
@binding(3) @group(0) var myTexture: texture_2d<f32>;

@stage(fragment)
fn main(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	// fout.color = fragment.color;

	// fout.color.x = fragment.uv.x;
	// fout.color.y = fragment.uv.y;
	// fout.color.a = 1.0;

	var uv = vec2<f32>(fragment.uv.x, 1.0 - fragment.uv.y);

	_ = mySampler;
	_ = myTexture;
	fout.color = textureSample(myTexture, mySampler, uv);

	if(fout.color.w < 1.0){
		discard;
	}

	
	var d = length(fragment.uv - 0.5);
	var dw = d * uniforms.size;
	dw = dw * dw;

	var viewPos = fragment.viewPos;
	viewPos.z = viewPos.z - dw;
	var projPos = uniforms.proj * vec4<f32>(viewPos, 1.0);
	fout.depth = projPos.z / projPos.w;

	return fout;
}
`;

let initialized = false;
let pipeline = null;
let uniformBuffer = null;
let bindGroup = null;
let sampler = null;

function createPipeline(renderer){

	let {device} = renderer;
	
	pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
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
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
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

	return pipeline;
}

function init(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	{
		pipeline = createPipeline(renderer);

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

	sampler = device.createSampler({
		magFilter: 'linear',
		minFilter: 'linear',
		mipmapFilter : 'linear',
		addressModeU: "clamp-to-edge",
		addressModeV: "clamp-to-edge",
		maxAnisotropy: 1,
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
		view.setFloat32(136, 20.0, true);
		view.setFloat32(140, 5, true);
		view.setFloat32(144, 5.0, true);
		view.setFloat32(148, 100.0, true);
	}

	renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
}

export function renderQuads(node, drawstate){

	if(!node.texture){
		return;
	}

	let {renderer} = drawstate;

	init(renderer);

	updateUniforms(drawstate);

	let {passEncoder} = drawstate.pass;

	passEncoder.setPipeline(pipeline);

	let bindGroup = renderer.device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 2, resource: sampler},
			{binding: 3, resource: node.texture.createView()},
		]
	});
	passEncoder.setBindGroup(0, bindGroup);

	let vboPosition = renderer.getGpuBuffer(node.positions);

	passEncoder.setVertexBuffer(0, vboPosition);

	let numVertices = node.positions.length / 3;
	passEncoder.draw(6, numVertices, 0, 0);
	

};