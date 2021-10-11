
import {Vector3, Matrix4, Geometry} from "potree";

const shaderSource = `
[[block]] struct Uniforms {
	worldView          : mat4x4<f32>;
	proj               : mat4x4<f32>;  // 128
	screen_width       : f32;
	screen_height      : f32;
	voxelGridSize      : f32;
	pad_0     : u32;         // 144
	bbMin              : vec3<f32>;    // 16     144
	bbMax              : vec3<f32>;    // 16     160
	pad_1              : u32;
};

struct Node{
	childMask         : u32;
	childOffset       : u32;
	level             : u32;
	pad_0         : u32;
	pad_1    : u32;
	pad_2      : u32;
	isLeaf            : u32;
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

struct LOD {
	depth     : u32;
	processed : bool;
	isLeaf    : bool;
	npos      : vec3<f32>;
};

[[block]] struct U32s {values : [[stride(4)]] array<u32>;};
[[block]] struct F32s {values : [[stride(4)]] array<f32>;};

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

[[block]] struct Voxels { values : [[stride(32)]] array<Voxel>; };
[[block]] struct Nodes{ values : [[stride(32)]] array<Node>; };

[[binding(0), group(0)]] var<uniform> uniforms         : Uniforms;
[[binding(1), group(0)]] var<storage, read> positions  : F32s;
[[binding(2), group(0)]] var<storage, read> colors     : U32s;
[[binding(3), group(0)]] var<storage, read> nodes      : Nodes;

struct VertexIn{
	[[builtin(vertex_index)]] index : u32;
	[[builtin(instance_index)]] instance_index : u32;
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
		index = index | 4u;
	}

	if(npos.y >= 0.5){
		index = index | 2u;
	}

	if(npos.z >= 0.5){
		index = index | 1u;
	}

	return index;
}

fn getLOD(position : vec3<f32>) -> LOD {

	var cubeSize = uniforms.bbMax.x - uniforms.bbMin.x;
	var npos = (position - uniforms.bbMin) / cubeSize;

	var nodeIndex = 0u;
	var depth = 0u;
	var node : Node;

	var result = LOD();

	for(var i = 0u; i < 20u; i = i + 1u){

		node = nodes.values[nodeIndex];
		var childIndex = toChildIndex(npos);

		var hasChild = (node.childMask & (1u << childIndex)) != 0u;

		// result.npos = vec3<f32>(f32(childIndex) / 7.0, 0.0, 0.0);

		// if(i == 1u){

		// 	// if(node.childMask == 2u){
		// 	if(nodeIndex == 1u){
		// 		result.npos = vec3<f32>(0.0, 1.0, 0.0);
		// 	}

		// 	break;
		// }

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
			// if(i == 1u && node.childMask == 2u){
			// 	result.npos = npos;
			// 	result.npos = vec3<f32>(1.0, 0.0, 0.0);
			// }
			break;
		}
	}

	
	result.depth = depth;
	result.isLeaf = node.isLeaf != 0u;

	return result;
}

fn doIgnore(){
	ignore(uniforms);
	ignore(positions);
	ignore(colors);
}

[[stage(vertex)]]
fn main_vertex(vertex : VertexIn) -> VertexOut {

	doIgnore();

	var node = nodes.values[vertex.instance_index];
	let cubeVertexIndex : u32 = vertex.index % 36u;
	var cubeOffset : vec3<f32> = CUBE_POS[cubeVertexIndex];
	var voxelIndex = vertex.index / 36u;

	var position = vec3<f32>(
		positions.values[3u * voxelIndex + 0u],
		positions.values[3u * voxelIndex + 1u],
		positions.values[3u * voxelIndex + 2u],
	);

	var lod = getLOD(position);

	var cubeSize = uniforms.bbMax.x - uniforms.bbMin.x;
	var chunkSize = cubeSize / pow(2.0, f32(node.level));
	var voxelSize = chunkSize / uniforms.voxelGridSize;
	var viewPos : vec4<f32> = uniforms.worldView * vec4<f32>(position + voxelSize * cubeOffset + voxelSize / 2.0, 1.0);
	var projPos : vec4<f32> = uniforms.proj * viewPos;

	var vout : VertexOut;

	vout.position = projPos;
	vout.color = vec4<f32>(
		f32((colors.values[voxelIndex] >>  0u) & 0xFFu) / 255.0,
		f32((colors.values[voxelIndex] >>  8u) & 0xFFu) / 255.0,
		f32((colors.values[voxelIndex] >> 16u) & 0xFFu) / 255.0,
		1.0);

	if(lod.depth != node.level){
		// discard!

		vout.position = vec4<f32>(10.0, 10.0, 10.0, 1.0);
	}

	// if(lod.isLeaf){
	// 	var blue = vec4<f32>(0.0, 0.0, 1.0, 1.0);
	// 	vout.color = 0.5 * vout.color + 0.5 * blue;
	// 	vout.color.w = 1.0;

	// 	vout.position = vec4<f32>(10.0, 10.0, 10.0, 1.0);
	// }

	// var gradientColor = GRADIENT[lod.depth] / 255.0;
	// vout.color = vec4<f32>(gradientColor, 1.0);

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
	}else{
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

		let maxNodes = 10_000;
		let nodeBufferHost = new ArrayBuffer(maxNodes * 16);
		let nodeBuffer = device.createBuffer({
			size: nodeBufferHost.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		let uniformBuffer = device.createBuffer({
			size: 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let state = {
			pipeline, nodeBufferHost, nodeBuffer, uniformBuffer,
		};

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
		view.setFloat32(144, box.min.x, true);
		view.setFloat32(148, box.min.y, true);
		view.setFloat32(152, box.min.z, true);
		view.setFloat32(160, box.max.x, true);
		view.setFloat32(164, box.max.y, true);
		view.setFloat32(168, box.max.z, true);

		view.setFloat32(128, size.width, true);
		view.setFloat32(132, size.height, true);
		view.setFloat32(136, node.voxelGridSize, true);
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

export function renderVoxelsLOD(root, drawstate){

	let {renderer, camera} = drawstate;
	let {passEncoder} = drawstate.pass;

	let state = getState(renderer, root);

	// let numVoxels = 0;
	let nodes = [];
	root.traverse((node) => {

		if(node.visible){
			nodes.push(node);
			// numVoxels += node.voxels.numVoxels;
		}
	});

	// sort breadth-first
	//nodes.sort((a, b) => a.id.localeCompare(b.id));
	nodes.sort((a, b) => {
		if(a.name.length !== b.name.length){
			return a.name.length - b.name.length;
		}else{
			return a.name.localeCompare(b.name);
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

		let view = new DataView(state.nodeBufferHost);
		for(let i = 0; i < nodes.length; i++){
			let node = nodes[i];

			let mask = childMaskOf(node);
			let childOffset = node.childOffset ?? 0;

			view.setUint32(32 * i + 0, mask, true);
			view.setUint32(32 * i + 4, childOffset, true);
			view.setUint32(32 * i + 8, node.level, true);
			view.setUint32(32 * i + 12, 1, true);
			view.setUint32(32 * i + 16, 0, true);
			view.setUint32(32 * i + 20, 0, true);
			view.setUint32(32 * i + 24, node.isLeaf, true);
		}

		renderer.device.queue.writeBuffer(state.nodeBuffer, 0, state.nodeBufferHost, 0, 32 * nodes.length);
	}

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
				{binding: 3, resource: {buffer: state.nodeBuffer}},
			],
		});
		passEncoder.setBindGroup(0, bindGroup);

		passEncoder.draw(36 * node.voxels.numVoxels, 1, 0, instanceIndex);

		instanceIndex++;
	}


}