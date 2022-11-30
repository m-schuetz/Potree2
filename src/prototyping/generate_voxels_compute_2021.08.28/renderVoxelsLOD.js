
import {Vector3, Matrix4, Geometry} from "potree";
import { voxelGridSize } from "./common.js";

const shaderSource = `
struct Uniforms {
	worldView          : mat4x4<f32>;
	proj               : mat4x4<f32>;  // 128
	screen_width       : f32;
	screen_height      : f32;
	voxelGridSize      : f32;
	// pad_1              : u32;         // 144
	voxelBaseIndex     : u32;         // 144
	bbMin              : vec3<f32>;    // 16     144
	bbMax              : vec3<f32>;    // 16     160
	pad_0              : u32;
};

struct Node{
	childMask         : u32;
	childOffset       : u32;
	level             : u32;
	processed         : u32;
	voxelBaseIndex    : u32;
};

struct Voxel {
	x     : f32;
	y     : f32;
	z     : f32;
	r     : u32;
	g     : u32;
	b     : u32;
	count : u32;
	pad1  : f32;
};

struct Voxels { values : array<Voxel> };
struct Nodes{ values : array<Node> };

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

var<private> GRADIENT : array<vec3<f32>, 4> = array<vec3<f32>, 4>(
	vec3<f32>(215.0,  25.0,  28.0),
	vec3<f32>(253.0, 174.0,  97.0),
	vec3<f32>(171.0, 221.0, 164.0),
	vec3<f32>( 43.0, 131.0, 186.0),
);

@binding(0) @group(0) var<uniform> uniforms         : Uniforms;
@binding(2) @group(0) var<storage, read> voxels     : Voxels;
@binding(3) @group(0) var<storage, read> nodes : Nodes;

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

fn toChildIndex(npos : vec3<f32>) -> u32 {

	var index = 0u;

	if(npos.x >= 0.5){
		index = index | 1u;
	}

	if(npos.y >= 0.5){
		index = index | 2u;
	}

	if(npos.z >= 0.5){
		index = index | 4u;
	}

	return index;
}

fn getLOD(position : vec3<f32>) -> u32 {

	var cubeSize = uniforms.bbMax.x - uniforms.bbMin.x;
	var npos = (position - uniforms.bbMin) / cubeSize;

	var nodeIndex = 0u;
	var depth = 0u;
	var processed = true;

	for(var i = 0u; i < 20u; i = i + 1u){

		var node = nodes.values[nodeIndex];
		var childIndex = toChildIndex(npos);
		if(node.processed == 0u){
			processed = false;
		}

		var hasChild = (node.childMask & (1u << childIndex)) != 0u;

		if(hasChild){

			depth = depth + 1u;

			var offsetMask = 0xFFu >> (8u - childIndex);
			var bitcount = countOneBits(offsetMask & node.childMask);

			nodeIndex = node.childOffset + bitcount;

			if(npos.x >= 0.5){
				npos.x = npos.x - 0.5;
			}
			if(npos.y >= 0.5){
				npos.y = npos.y - 0.5;
			}
			if(npos.z >= 0.5){
				npos.z = npos.z - 0.5;
			}
			npos = npos * 2.0;

		}else{
			break;
		}
	}

	if(!processed){
		depth = 100u;
	}

	return depth;
}

fn doIgnore(){
	ignore(uniforms);
	var a10 = voxels.values[0];
	var a20 = nodes.values[0];
}

@vertex
fn main_vertex(vertex : VertexIn) -> VertexOut {

	doIgnore();

	var node = nodes.values[vertex.instance_index];
	let cubeVertexIndex : u32 = vertex.index % 36u;
	var cubeOffset : vec3<f32> = CUBE_POS[cubeVertexIndex];
	var voxelIndex = vertex.index / 36u + node.voxelBaseIndex;
	var voxel = voxels.values[voxelIndex];

	var position = vec3<f32>(
		voxel.x, 
		voxel.y, 
		voxel.z, 
	);

	var lod = getLOD(position);

	var cubeSize = uniforms.bbMax.x - uniforms.bbMin.x;
	var chunkSize = cubeSize / pow(2.0, f32(node.level));
	var voxelSize = chunkSize / uniforms.voxelGridSize;
	var viewPos : vec4<f32> = uniforms.worldView * vec4<f32>(position + voxelSize * cubeOffset, 1.0);
	var projPos : vec4<f32> = uniforms.proj * viewPos;

	var vout : VertexOut;

	vout.position = projPos;
	vout.color = vec4<f32>(
		f32(voxel.r) / 255.0, 
		f32(voxel.g) / 255.0, 
		f32(voxel.b) / 255.0, 
		1.0);

	// var gradientColor = GRADIENT[lod] / 255.0;
	// vout.color = vec4<f32>(gradientColor, 1.0);

	if(lod == 100u){
		vout.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
	}else if(lod != node.level){
		// discard!

		vout.position = vec4<f32>(10.0, 10.0, 10.0, 1.0);
	}

	return vout;
}

@fragment
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	fout.color = fragment.color;

	return fout;
}
`;

let initialized = false;
let pipeline = null;
let nodeBuffer = null;
let nodeBufferHost = null;
let uniformBuffer = null;

function init(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	pipeline = device.createRenderPipeline({
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

	let maxNodes = 10_000;
	nodeBufferHost = new ArrayBuffer(maxNodes * 16);
	nodeBuffer = device.createBuffer({
		size: nodeBufferHost.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});

	uniformBuffer = device.createBuffer({
		size: 256,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	initialized = true;
}

function updateUniforms(renderer, root){

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

		let box = root.boundingBox;
		view.setFloat32(144, box.min.x, true);
		view.setFloat32(148, box.min.y, true);
		view.setFloat32(152, box.min.z, true);
		view.setFloat32(160, box.max.x, true);
		view.setFloat32(164, box.max.y, true);
		view.setFloat32(168, box.max.z, true);

		view.setFloat32(128, size.width, true);
		view.setFloat32(132, size.height, true);
		view.setFloat32(136, voxelGridSize, true);
	}

	renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
	
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

export function renderVoxelsLOD(root, drawstate){

	let {renderer, camera} = drawstate;
	let {passEncoder} = drawstate.pass;

	init(renderer);

	let numVoxels = 0;
	let nodes = [];
	root.traverse((node) => {

		if(node.visible){
			nodes.push(node);
			numVoxels += node.voxels.numVoxels;
		}
	});
	window.nodes = nodes;

	// sort breadth-first
	//nodes.sort((a, b) => a.id.localeCompare(b.id));
	nodes.sort((a, b) => {
		if(a.id.length !== b.id.length){
			return a.id.length - b.id.length;
		}else{
			return a.id.localeCompare(b.id);
		}
	});

	{

		for(let i = 0; i < nodes.length; i++){
			let node = nodes[i];

			if(node.parent !== null){
				node.parent.childOffset = Infinity;
			}
		}

		for(let i = 0; i < nodes.length; i++){
			let node = nodes[i];

			if(node.parent !== null){
				node.parent.childOffset = Math.min(node.parent.childOffset, i);
			}
		}

		let view = new DataView(nodeBufferHost);
		for(let i = 0; i < nodes.length; i++){
			let node = nodes[i];

			let mask = childMaskOf(node);
			let childOffset = node.childOffset ?? 0;

			view.setUint32(32 * i + 0, mask, true);
			view.setUint32(32 * i + 4, childOffset, true);
			view.setUint32(32 * i + 8, node.level, true);
			view.setUint32(32 * i + 12, node.processed ? 1 : 0, true);
			view.setUint32(32 * i + 16, node.voxels.firstVoxel, true);

		}

		renderer.device.queue.writeBuffer(nodeBuffer, 0, nodeBufferHost, 0, 32 * nodes.length);
	}

	updateUniforms(renderer, root);

	passEncoder.setPipeline(pipeline);

	let bindGroup = renderer.device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 2, resource: {buffer: nodes[0].voxels.gpu_voxels}},
			{binding: 3, resource: {buffer: nodeBuffer}},
		],
	});
	passEncoder.setBindGroup(0, bindGroup);

	let instanceIndex = 0;
	for(let node of nodes){

		passEncoder.draw(36 * node.voxels.numVoxels, 1, 0, instanceIndex);

		instanceIndex++;
	}


}