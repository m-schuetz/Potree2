
import {Vector3, Matrix4, Geometry} from "potree";
import { voxelGridSize } from "./common.js";

const shaderSource = `
[[block]] struct Uniforms {
	worldView          : mat4x4<f32>;
	proj               : mat4x4<f32>;  // 128
	screen_width       : f32;
	screen_height      : f32;
	voxelSize          : f32;
	voxelBaseIndex     : u32;         // 144
	bbMin              : vec3<f32>;    // 16     144
	bbMax              : vec3<f32>;    // 16     160
	pad_0              : u32;
	level              : u32;
};

struct Node{
	childMask         : u32;
	childOffset       : u32;
	level             : u32;
	pad2              : u32;
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

[[block]] struct Voxels { values : [[stride(32)]] array<Voxel>; };
[[block]] struct Nodes{ values : [[stride(16)]] array<Node>; };

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

[[binding(0), group(0)]] var<uniform> uniforms         : Uniforms;
[[binding(2), group(0)]] var<storage, read> voxels     : Voxels;
[[binding(3), group(0)]] var<storage, read> nodes : Nodes;

struct VertexIn{
	[[builtin(vertex_index)]] index : u32;
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

	for(var i = 0u; i < 20u; i = i + 1u){

		var node = nodes.values[nodeIndex];
		var childIndex = toChildIndex(npos);

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

	return depth;
}

fn doIgnore(){
	ignore(uniforms);
	var a10 = voxels.values[0];
	var a20 = nodes.values[0];
}

[[stage(vertex)]]
fn main_vertex(vertex : VertexIn) -> VertexOut {

	doIgnore();

	let cubeVertexIndex : u32 = vertex.index % 36u;
	var cubeOffset : vec3<f32> = CUBE_POS[cubeVertexIndex];
	var voxelIndex = vertex.index / 36u + uniforms.voxelBaseIndex;
	// var voxelIndex = 1u;
	var voxel = voxels.values[voxelIndex];

	var position = vec3<f32>(
		voxel.x, 
		voxel.y, 
		voxel.z, 
	);

	var lod = getLOD(position);

	var viewPos : vec4<f32> = uniforms.worldView * vec4<f32>(position + uniforms.voxelSize * cubeOffset, 1.0);
	var projPos : vec4<f32> = uniforms.proj * viewPos;

	var vout : VertexOut;

	vout.position = projPos;
	vout.color = vec4<f32>(
		f32(voxel.r) / 255.0, 
		f32(voxel.g) / 255.0, 
		f32(voxel.b) / 255.0, 
		1.0);

	// vout.color = vec4<f32>(
	// 	f32(lod) / 2.0,
	// 	0.0, 0.0, 1.0
	// );

	if(lod != uniforms.level){
		// discard!

		vout.position = vec4<f32>(10.0, 10.0, 10.0, 1.0);
	}

	return vout;
}

[[stage(fragment)]]
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
	}

	let {device} = renderer;

	let uniformBuffer = device.createBuffer({
		size: 256,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	// let bindGroup = device.createBindGroup({
	// 	layout: pipeline.getBindGroupLayout(0),
	// 	entries: [
	// 		{binding: 0, resource: {buffer: uniformBuffer}},
	// 		{binding: 2, resource: {buffer: node.voxels.gpu_voxels}},
	// 	],
	// });

	let state = {uniformBuffer};

	stateCache.set(node, state);

	return state;
}

let initialized = false;
let pipeline = null;
let nodeBuffer = null;
let nodeBufferHost = null;

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

	initialized = true;
}

function updateUniforms(renderer, root, node){
	
	let {uniformBuffer} = getState(renderer, node);

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

		let chunkSize = node.boundingBox.max.x - node.boundingBox.min.x;
		let voxelSize = chunkSize / voxelGridSize;
		view.setFloat32(128, size.width, true);
		view.setFloat32(132, size.height, true);
		view.setFloat32(136, voxelSize, true);
		view.setUint32(140, node.voxels.firstVoxel, true);
		view.setUint32(176, node.level, true);
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
	nodes.sort((a, b) => a.id.localeCompare(b.id));

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

			view.setUint32(16 * i + 0, mask, true);
			view.setUint32(16 * i + 4, childOffset, true);
			view.setUint32(16 * i + 8, node.level, true);

		}

		renderer.device.queue.writeBuffer(nodeBuffer, 0, nodeBufferHost, 0, 16 * nodes.length);
	}

	for(let node of nodes){
		updateUniforms(renderer, root, node);
	}

	passEncoder.setPipeline(pipeline);

	for(let node of nodes){
		// let {bindGroup} = getState(renderer, node);

		// passEncoder.setBindGroup(0, bindGroup);

		let {uniformBuffer} = getState(renderer, node);
		let bindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 2, resource: {buffer: node.voxels.gpu_voxels}},
				{binding: 3, resource: {buffer: nodeBuffer}},
			],
		});
		passEncoder.setBindGroup(0, bindGroup);

		passEncoder.draw(36 * node.voxels.numVoxels, 1, 0, 0);
		// passEncoder.draw(36 * node.voxels.numVoxels, 1, 0, 0);
	}


}