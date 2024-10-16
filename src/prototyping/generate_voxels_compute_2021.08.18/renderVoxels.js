
import {Vector3, Matrix4, Geometry} from "potree";

const shaderSource = `
struct Uniforms {
	worldView      : mat4x4<f32>;
	proj           : mat4x4<f32>;
	screen_width   : f32;
	screen_height  : f32;
	size           : f32;

};

struct Metadata {
	offsetCounter   : u32;
	numVoxelsAdded  : u32;
	pad1 : u32;
	pad2 : u32;
	value0 : u32;
	value1 : u32;
	value2 : u32;
	value3 : u32;
	value_f32_0 : f32;
	value_f32_1 : f32;
	value_f32_2 : f32;
	value_f32_3 : f32;
};

struct Voxel{
	x      : f32;
	y      : f32;
	z      : f32;
	r      : u32;
	g      : u32;
	b      : u32;
	count  : u32;
	size   : f32;
};

struct Voxels { values : array<Voxel> };

var<private> CUBE_POS : array<vec3<f32>, 36> = array<vec3<f32>, 36>(
	vec3<f32>(-0.5, -0.5, -0.5),
	vec3<f32>(0.5,  0.5, -0.5),
	vec3<f32>(0.5, -0.5, -0.5),
	vec3<f32>(-0.5, -0.5, -0.5),
	vec3<f32>(-0.5,  0.5, -0.5),
	vec3<f32>(0.5,  0.5, -0.5),
	vec3<f32>(-0.5, -0.5,  0.5),
	vec3<f32>(0.5, -0.5,  0.5),
	vec3<f32>(0.5,  0.5,  0.5),
	vec3<f32>(-0.5, -0.5,  0.5),
	vec3<f32>(0.5,  0.5,  0.5),
	vec3<f32>(-0.5,  0.5,  0.5),
	vec3<f32>(-0.5, -0.5, -0.5,),
	vec3<f32>(-0.5,  0.5,  0.5,),
	vec3<f32>(-0.5,  0.5, -0.5,),
	vec3<f32>(-0.5, -0.5, -0.5,),
	vec3<f32>(-0.5, -0.5,  0.5,),
	vec3<f32>(-0.5,  0.5,  0.5,),
	vec3<f32>(0.5, -0.5, -0.5),
	vec3<f32>(0.5,  0.5, -0.5),
	vec3<f32>(0.5,  0.5,  0.5),
	vec3<f32>(0.5, -0.5, -0.5),
	vec3<f32>(0.5,  0.5,  0.5),
	vec3<f32>(0.5, -0.5,  0.5),
	vec3<f32>(-0.5, 0.5, -0.5),
	vec3<f32>(0.5, 0.5,  0.5),
	vec3<f32>(0.5, 0.5, -0.5),
	vec3<f32>(-0.5, 0.5, -0.5),
	vec3<f32>(-0.5, 0.5,  0.5),
	vec3<f32>(0.5, 0.5,  0.5),
	vec3<f32>(-0.5, -0.5, -0.5),
	vec3<f32>(0.5, -0.5, -0.5),
	vec3<f32>(0.5, -0.5,  0.5),
	vec3<f32>(-0.5, -0.5, -0.5),
	vec3<f32>(0.5, -0.5,  0.5),
	vec3<f32>(-0.5, -0.5,  0.5),
);

@binding(0) @group(0) var<uniform> uniforms         : Uniforms;
@binding(1) @group(0) var<storage, read> metadata   : Metadata;
@binding(2) @group(0) var<storage, read> voxels     : Voxels;

struct VertexIn{
	@builtin(vertex_index) index : u32,
};

struct VertexOut{
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec4<f32>,
};

struct FragmentIn{
	@location(0) color : vec4<f32>,
};

struct FragmentOut{
	@location(0) color : vec4<f32>,
};

fn doIgnore(){
	ignore(uniforms);
	ignore(metadata);
	var a10 = voxels.values[0];
}

@vertex
fn main_vertex(vertex : VertexIn) -> VertexOut {

	doIgnore();

	let cubeVertexIndex : u32 = vertex.index % 36u;
	var cubeOffset : vec3<f32> = CUBE_POS[cubeVertexIndex];
	var voxelIndex = vertex.index / 36u;
	var voxel = voxels.values[voxelIndex];

	var position = vec3<f32>(
		voxel.x, 
		voxel.y, 
		voxel.z, 
	);

	var viewPos : vec4<f32> = uniforms.worldView * vec4<f32>(position + voxel.size * cubeOffset, 1.0);
	var projPos : vec4<f32> = uniforms.proj * viewPos;

	var vout : VertexOut;

	vout.position = projPos;
	vout.color = vec4<f32>(
		f32(voxel.r) / 255.0, 
		f32(voxel.g) / 255.0, 
		f32(voxel.b) / 255.0, 
		1.0);

	return vout;
}

@fragment
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	fout.color = fragment.color;

	return fout;
}
`;

let stateCache = new Map();
function getState(renderer, voxels){

	if(stateCache.has(voxels.gpu_meta)){
		return stateCache.get(voxels.gpu_meta);
	}

	let {device} = renderer;

	let pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: shaderSource}),
			entryPoint: "main_vertex",
			buffers: []
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

	let uniformBuffer = renderer.device.createBuffer({
		size: 256,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	let bindGroup = renderer.device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: {buffer: voxels.gpu_meta}},
			{binding: 2, resource: {buffer: voxels.gpu_voxels}},
		],
	});

	let state = {pipeline, uniformBuffer, bindGroup};

	stateCache.set(voxels.gpu_meta, state);

	return state;
}


export function renderVoxels(drawstate, voxels){

	let {renderer, camera} = drawstate;
	let {device} = renderer;
	let {passEncoder} = drawstate.pass;

	let {pipeline, uniformBuffer, bindGroup} = getState(renderer, voxels);

	passEncoder.setBindGroup(0, bindGroup);

	{

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
			view.setFloat32(136, 0.1, true);
		}

		renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
	}

	passEncoder.setPipeline(pipeline);

	passEncoder.draw(36 * voxels.numVoxels, 1, 0, 0);


}