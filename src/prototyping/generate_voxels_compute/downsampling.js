
import { Vector3 } from "../../math/Vector3.js";
import {Chunk, voxelGridSize, chunkGridSize, toIndex1D, toIndex3D} from "./common.js";
import {storage_flags, uniform_flags} from "./common.js";
import { generateVoxelsCompute } from "./generate_voxels_compute.js";
import {renderVoxels} from "./renderVoxels.js";
import {SPECTRAL} from "potree";

export let csVoxelizing = `


[[block]] struct Uniforms {
	numTriangles       : u32;
	firstTriangle      : u32;
	chunkGridSize      : u32;
	voxelGridSize      : u32;
	chunkMin           : vec4<f32>;      // offset(16)
	chunkMax           : vec4<f32>;      // offset(32)
	nodeMin            : vec4<f32>;      // offset(48)
	nodeMax            : vec4<f32>;      // offset(64)
};

[[block]] struct Metadata {
	offsetCounter : atomic<u32>;
	pad0 : u32;
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

[[block]] struct F32s { values : [[stride(4)]] array<f32>; };
[[block]] struct U32s { values : [[stride(4)]] array<u32>; };
[[block]] struct I32s { values : [[stride(4)]] array<i32>; };
[[block]] struct AU32s { values : [[stride(4)]] array<atomic<u32>>; };
[[block]] struct AI32s { values : [[stride(4)]] array<atomic<i32>>; };

// IN
[[binding( 0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(10), group(0)]] var<storage, read_write> indices : U32s;
[[binding(11), group(0)]] var<storage, read_write> positions : F32s;
[[binding(12), group(0)]] var<storage, read_write> colors : U32s;

// OUT
[[binding(20), group(0)]] var<storage, read_write> tricountGrid : AU32s;
[[binding(21), group(0)]] var<storage, read_write> LutGrid : AU32s;
[[binding(22), group(0)]] var<storage, read_write> voxelGrid : AU32s;
[[binding(23), group(0)]] var<storage, read_write> sortedIndices : U32s;

[[binding(50), group(0)]] var<storage, read_write> metadata : Metadata;

fn toVoxelPos(gridSize : u32, position : vec3<f32>) -> vec3<f32>{

	var bbMin = vec3<f32>(uniforms.chunkMin.x, uniforms.chunkMin.y, uniforms.chunkMin.z);
	var bbMax = vec3<f32>(uniforms.chunkMax.x, uniforms.chunkMax.y, uniforms.chunkMax.z);
	var bbSize = bbMax - bbMin;
	var cubeSize = max(max(bbSize.x, bbSize.y), bbSize.z);
	var fGridSize = f32(gridSize);

	var gx = fGridSize * (position.x - uniforms.chunkMin.x) / cubeSize;
	var gy = fGridSize * (position.y - uniforms.chunkMin.y) / cubeSize;
	var gz = fGridSize * (position.z - uniforms.chunkMin.z) / cubeSize;

	return vec3<f32>(gx, gy, gz);
}

fn toIndex1D(gridSize : u32, voxelPos : vec3<f32>) -> u32{

	var icoord = vec3<u32>(voxelPos);

	return icoord.x 
		+ gridSize * icoord.y 
		+ gridSize * gridSize * icoord.z;
}

fn toIndex3D(gridSize : u32, index : u32) -> vec3<u32>{
	var z = index / (gridSize * gridSize);
	var y = (index - gridSize * gridSize * z) / gridSize;
	var x = index % gridSize;

	return vec3<u32>(x, y, z);
}

fn loadPosition(vertexIndex : u32) -> vec3<f32> {
	
	var position = vec3<f32>(
		positions.values[3u * vertexIndex + 0u],
		positions.values[3u * vertexIndex + 1u],
		positions.values[3u * vertexIndex + 2u],
	);

	return position;
};

fn loadColor(vertexIndex : u32) -> vec3<u32> {
	
	var uColor = colors.values[vertexIndex];
	var R = (uColor >>  0u) & 0xFFu;
	var G = (uColor >>  8u) & 0xFFu;
	var B = (uColor >> 16u) & 0xFFu;

	var color = vec3<u32>(R, G, B);

	return color;
};

fn doIgnore(){
	
	ignore(uniforms);

	var a00 = indices.values[0];
	var a10 = positions.values[0];
	var a20 = colors.values[0];

	var a30 = atomicLoad(&tricountGrid.values[0]);
	var a40 = atomicLoad(&LutGrid.values[0]);
	var a50 = atomicLoad(&voxelGrid.values[0]);
	var a60 = sortedIndices.values[0];

	ignore(metadata);
	
}

[[stage(compute), workgroup_size(128)]]
fn main_accumulate([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	doIgnore();

	var triangleIndex = GlobalInvocationID.x;

	if(triangleIndex >= uniforms.numTriangles){
		return;
	}


	var i0 = indices.values[3u * triangleIndex + 0u];
	var i1 = indices.values[3u * triangleIndex + 1u];
	var i2 = indices.values[3u * triangleIndex + 2u];

	var p0 = loadPosition(i0);
	var p1 = loadPosition(i1);
	var p2 = loadPosition(i2);
	var center = (p0 + p1 + p2) / 3.0;

	var voxelPos = toVoxelPos(uniforms.voxelGridSize, center);
	var voxelIndex = toIndex1D(uniforms.voxelGridSize, voxelPos);
	var chunkPos = toVoxelPos(uniforms.chunkGridSize, center);
	var chunkIndex = toIndex1D(uniforms.chunkGridSize, chunkPos);

	{ // accumulate voxels
		var color = (loadColor(i0) + loadColor(i1) + loadColor(i2)) / 3u;

		var a00 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 0u], color.x);
		var a10 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 1u], color.y);
		var a20 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 2u], color.z);
		var a30 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 3u], 1u);
	}

	{ // accumulate triangle counts
		var a00 = atomicAdd(&tricountGrid.values[chunkIndex], 1u);
	}
}

// [[stage(compute), workgroup_size(128)]]
// fn main_gatherVoxels([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

// 	var voxelIndex = GlobalInvocationID.x;

// 	doIgnore();

// 	var maxVoxels = uniforms.voxelGridSize * uniforms.voxelGridSize * uniforms.voxelGridSize;
// 	if(voxelIndex >= maxVoxels){
// 		return;
// 	}


// }



[[stage(compute), workgroup_size(128)]]
fn main_test([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	var triangleIndex = GlobalInvocationID.x;

	atomicStore(&metadata.offsetCounter, 2355u);

	doIgnore();

	if(triangleIndex >= uniforms.numTriangles){
		return;
	}


}

`;

let gpu_voxelGrid = null;
let gpu_tricountGrid = null;
let gpu_LutGrid = null;
let gpu_sortedIndices = null;
let gpu_metadata = null;


function initialize(renderer, node){

	let {device} = renderer;

	let tricountGridSize = 4 * chunkGridSize ** 3;
	let voxelGridByteSize = 4 * voxelGridSize ** 3;

	gpu_tricountGrid = device.createBuffer({size: tricountGridSize, usage: storage_flags});
	gpu_LutGrid = device.createBuffer({size: tricountGridSize, usage: storage_flags});
	gpu_voxelGrid = device.createBuffer({size: voxelGridByteSize, usage: storage_flags});
	gpu_metadata = device.createBuffer({size: 256, usage: storage_flags});

}


async function processChunk(renderer, node, chunk){

	let {device} = renderer;

	let uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});
	{
		let buffer = new ArrayBuffer(256);
		let view = new DataView(buffer);

		view.setUint32( 0, chunk.triangles.numIndices / 3, true);
		view.setUint32( 4, chunk.triangles.firstIndex / 3, true);
		view.setUint32( 8, chunkGridSize, true);
		view.setUint32(16, voxelGridSize, true);

		// chunk box
		view.setFloat32(16 +  0, chunk.boundingBox.min.x, true);
		view.setFloat32(16 +  4, chunk.boundingBox.min.y, true);
		view.setFloat32(16 +  8, chunk.boundingBox.min.z, true);
		view.setFloat32(16 + 16, chunk.boundingBox.max.x, true);
		view.setFloat32(16 + 20, chunk.boundingBox.max.y, true);
		view.setFloat32(16 + 24, chunk.boundingBox.max.z, true);

		// node box
		let nodeCube = node.boundingBox.cube();
		view.setFloat32(48 +  0, nodeCube.min.x, true);
		view.setFloat32(48 +  4, nodeCube.min.y, true);
		view.setFloat32(48 +  8, nodeCube.min.z, true);
		view.setFloat32(48 + 16, nodeCube.max.x, true);
		view.setFloat32(48 + 20, nodeCube.max.y, true);
		view.setFloat32(48 + 24, nodeCube.max.z, true);

		device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
	}

	let gpu_sortedIndices = device.createBuffer({size: 4 * chunk.triangles.numIndices, usage: storage_flags});

	let bindGroups = [
		{
			location: 0,
			entries: [
				{binding:  0, resource: {buffer: uniformBuffer}},
				{binding: 10, resource: {buffer: chunk.triangles.gpu_indices}},
				{binding: 11, resource: {buffer: chunk.triangles.gpu_position}},
				{binding: 12, resource: {buffer: chunk.triangles.gpu_color}},

				{binding: 20, resource: {buffer: gpu_tricountGrid}},
				{binding: 21, resource: {buffer: gpu_LutGrid}},
				{binding: 22, resource: {buffer: gpu_voxelGrid}},
				{binding: 23, resource: {buffer: gpu_sortedIndices}},

				{binding: 50, resource: {buffer: gpu_metadata}},
			],
		}
	];

	let numTriangles = chunk.triangles.numIndices / 3;
	renderer.runCompute({
		code: csVoxelizing,
		entryPoint: "main_accumulate",
		bindGroups: bindGroups,
		dispatchGroups: [Math.ceil(numTriangles / 128)],
	});

	// renderer.runCompute({
	// 	code: csVoxelizing,
	// 	entryPoint: "main_gatherVoxels",
	// 	bindGroups: bindGroups,
	// 	dispatchGroups: [Math.ceil((voxelGridSize ** 3) / 128)],
	// });

	// let adg = await renderer.readBuffer(gpu_metadata, 0, 4);
	// console.log(new Uint32Array(adg))

	let buffer = await renderer.readBuffer(gpu_tricountGrid, 0, 4 * chunkGridSize ** 3);
	let u32 = new Uint32Array(buffer);
	console.log(u32);

}

export async function doDownsampling(renderer, node){

	initialize(renderer, node);

	let root = new Chunk();
	root.boundingBox = node.boundingBox.cube();
	root.triangles = {
		gpu_position   : renderer.getGpuBuffer(node.geometry.findBuffer("position")),
		gpu_uv         : renderer.getGpuBuffer(node.geometry.findBuffer("uv")),
		gpu_color      : renderer.getGpuBuffer(node.geometry.findBuffer("color")),
		gpu_indices    : renderer.getGpuBuffer(node.geometry.indices),
		firstIndex     : 0,
		numIndices     : node.geometry.indices.length,
	};
	root.voxels = {
		gpu_voxels     : renderer.device.createBuffer({size: 32 * 4_000_000, usage: storage_flags}),
		firstVoxel     : 0,
		numVoxels      : 0,
	};

	processChunk(renderer, node, root);
	
}