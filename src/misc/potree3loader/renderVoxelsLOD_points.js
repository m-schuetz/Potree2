
import {Vector3, Matrix4, Geometry} from "potree";

const shaderSource = `
struct Uniforms {
	worldView          : mat4x4<f32>;
	proj               : mat4x4<f32>;  // 128
	screen_width       : f32;
	screen_height      : f32;
	voxelGridSize      : f32;
	voxelSize          : f32;
	bbMin              : vec3<f32>;    // 16     144
	bbMax              : vec3<f32>;    // 16     160
	point_size         : f32;
	pad_1              : u32;
};

struct U32s {values : array<u32>;};
struct F32s {values : array<f32>;};

var<private> GRADIENT : array<vec3<f32>, 4> = array<vec3<f32>, 4>(
	vec3<f32>(215.0,  25.0,  28.0),
	vec3<f32>(253.0, 174.0,  97.0),
	vec3<f32>(171.0, 221.0, 164.0),
	vec3<f32>( 43.0, 131.0, 186.0),
);

@binding(0) @group(0) var<uniform> uniforms         : Uniforms;
@binding(1) @group(0) var<storage, read> positions  : F32s;
@binding(2) @group(0) var<storage, read> colors     : U32s;

struct VertexIn{
	@builtin(vertex_index) index : u32,
	@builtin(instance_index) instance_index : u32,
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
	_ = uniforms;
	_ = &positions;
	_ = &colors;
}

@stage(vertex)
fn main_vertex(vertex : VertexIn) -> VertexOut {

	doIgnore();

	var position = vec3<f32>(
		positions.values[3u * vertex.index + 0u],
		positions.values[3u * vertex.index + 1u],
		positions.values[3u * vertex.index + 2u],
	);

	var viewPos : vec4<f32> = uniforms.worldView * vec4<f32>(position, 1.0);
	var projPos : vec4<f32> = uniforms.proj * viewPos;

	var vout : VertexOut;

	vout.position = projPos;
	vout.color = vec4<f32>(
		f32((colors.values[vertex.index] >>  0u) & 0xFFu) / 255.0,
		f32((colors.values[vertex.index] >>  8u) & 0xFFu) / 255.0,
		f32((colors.values[vertex.index] >> 16u) & 0xFFu) / 255.0,
		1.0);

	return vout;
}

@stage(fragment)
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	fout.color = fragment.color;

	return fout;
}
`;


let stateCache = new Map();
function getState(renderer, node){

	if(stateCache.has(node)){
		return stateCache.get(node);
	}else{
		let {device} = renderer;

		let pipeline = device.createRenderPipeline({
			layout: "auto",
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
				topology: 'point-list',
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

		let state = {pipeline, uniformBuffer};

		stateCache.set(node, state);

		return state;
	}
}


function updateUniforms(renderer, node){

	let state = getState(renderer, node);

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

		let box = node.boundingBox;
		let voxelSize = (box.max.x - box.min.x) / node.voxelGridSize;

		view.setFloat32(144, box.min.x, true);
		view.setFloat32(148, box.min.y, true);
		view.setFloat32(152, box.min.z, true);
		view.setFloat32(160, box.max.x, true);
		view.setFloat32(164, box.max.y, true);
		view.setFloat32(168, box.max.z, true);

		view.setFloat32(128, size.width, true);
		view.setFloat32(132, size.height, true);
		view.setFloat32(136, node.voxelGridSize, true);
		view.setFloat32(140, voxelSize, true);

		view.setFloat32(176, 2.0, true);
	}

	renderer.device.queue.writeBuffer(state.uniformBuffer, 0, data, 0, data.byteLength);
	
}

function childMaskOf(node){

	let mask = 0;

	for(let i = 0; i < node.children.length; i++){
		if(node.children[i]?.visible){
			mask = mask | (1 << i);
		}

	}

	return mask;
}

export function renderVoxelsLOD_points(root, drawstate){

	let {renderer, camera} = drawstate;
	let {passEncoder} = drawstate.pass;

	let state = getState(renderer, root);

	let nodes = [];
	root.traverse((node) => {
		if(node.visible){
			nodes.push(node);
		}
	});

	updateUniforms(renderer, root);

	passEncoder.setPipeline(state.pipeline);

	let instanceIndex = 0;
	for(let node of nodes){

		if(!node.voxels){
			continue;
		}

		let vboPositions = renderer.getGpuBuffer(node.voxels.positions);
		let vboColors = renderer.getGpuBuffer(node.voxels.colors);

		let bindGroup = renderer.device.createBindGroup({
			layout: state.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: state.uniformBuffer}},
				{binding: 1, resource: {buffer: vboPositions}},
				{binding: 2, resource: {buffer: vboColors}},
			],
		});
		passEncoder.setBindGroup(0, bindGroup);

		passEncoder.draw(node.voxels.numVoxels, 1, 0, instanceIndex);

		instanceIndex++;
	}


}