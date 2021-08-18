
import { Vector3 } from "../../math/Vector3.js";
import {Chunk, voxelGridSize, chunkGridSize, toIndex1D, toIndex3D, computeChildBox} from "./common.js";
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
	tricountGridOffset : u32;            // offset(80)
	LutGridOffset      : u32;
	voxelGridOffset    : u32;
	chunkIndex         : u32;
	voxelBaseIndex     : u32;
};

struct Chunk {
	numVoxels       : atomic<u32>;
	numTriangles    : atomic<u32>;
	lutCounter      : atomic<u32>;
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

[[block]] struct F32s { values : [[stride(4)]] array<f32>; };
[[block]] struct U32s { values : [[stride(4)]] array<u32>; };
[[block]] struct I32s { values : [[stride(4)]] array<i32>; };
[[block]] struct AU32s { values : [[stride(4)]] array<atomic<u32>>; };
[[block]] struct AI32s { values : [[stride(4)]] array<atomic<i32>>; };
[[block]] struct Chunks { values : [[stride(32)]] array<Chunk>; };
[[block]] struct Voxels { values : [[stride(32)]] array<Voxel>; };

[[binding( 0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(10), group(0)]] var<storage, read_write> indices : U32s;
[[binding(11), group(0)]] var<storage, read_write> positions : F32s;
[[binding(12), group(0)]] var<storage, read_write> colors : U32s;

[[binding(20), group(0)]] var<storage, read_write> grids : AU32s;

[[binding(30), group(0)]] var<storage, read_write> chunks : Chunks;
[[binding(31), group(0)]] var<storage, read_write> voxels : Voxels;
[[binding(32), group(0)]] var<storage, read_write> sortedIndices : U32s;

// [[binding(50), group(0)]] var<storage, read_write> metadata : Metadata;

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

	var a30 = atomicLoad(&grids.values[0]);
	var a40 = atomicLoad(&chunks.values[0].numVoxels);
	var a50 = voxels.values[0];
	var a60 = sortedIndices.values[0];

	// ignore(metadata);
	
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

		
		var a00 = atomicAdd(&grids.values[uniforms.voxelGridOffset + 4u * voxelIndex + 0u], color.x);
		var a10 = atomicAdd(&grids.values[uniforms.voxelGridOffset + 4u * voxelIndex + 1u], color.y);
		var a20 = atomicAdd(&grids.values[uniforms.voxelGridOffset + 4u * voxelIndex + 2u], color.z);
		var a30 = atomicAdd(&grids.values[uniforms.voxelGridOffset + 4u * voxelIndex + 3u], 1u);
	}

	{ // accumulate triangle counts
		var a00 = atomicAdd(&grids.values[uniforms.tricountGridOffset + chunkIndex], 1u);
	}
}

[[stage(compute), workgroup_size(128)]]
fn main_gatherVoxels([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	var voxelIndex = GlobalInvocationID.x;

	doIgnore();

	var maxVoxels = uniforms.voxelGridSize * uniforms.voxelGridSize * uniforms.voxelGridSize;
	if(voxelIndex >= maxVoxels){
		return;
	}

	var count = atomicLoad(&grids.values[uniforms.voxelGridOffset + 4u * voxelIndex + 3u]);

	if(count == 0u){
		return;
	}

	var R = atomicLoad(&grids.values[uniforms.voxelGridOffset + 4u * voxelIndex + 0u]);
	var G = atomicLoad(&grids.values[uniforms.voxelGridOffset + 4u * voxelIndex + 1u]);
	var B = atomicLoad(&grids.values[uniforms.voxelGridOffset + 4u * voxelIndex + 2u]);

	var cubeSize = uniforms.chunkMax.x - uniforms.chunkMin.x;
	var voxelSize = cubeSize / f32(uniforms.voxelGridSize);
	var voxelIndex3D = toIndex3D(uniforms.voxelGridSize, voxelIndex);

	var localProcessed = atomicAdd(&chunks.values[uniforms.chunkIndex].numVoxels, 1u);

	var voxel = Voxel();
	voxel.x = cubeSize * f32(voxelIndex3D.x) / f32(uniforms.voxelGridSize) + uniforms.chunkMin.x + 0.5 * voxelSize;
	voxel.y = cubeSize * f32(voxelIndex3D.y) / f32(uniforms.voxelGridSize) + uniforms.chunkMin.y + 0.5 * voxelSize;
	voxel.z = cubeSize * f32(voxelIndex3D.z) / f32(uniforms.voxelGridSize) + uniforms.chunkMin.z + 0.5 * voxelSize;
	voxel.r = R / count;
	voxel.g = G / count;
	voxel.b = B / count;
	voxel.count = count;

	var outIndex = uniforms.voxelBaseIndex + localProcessed;
	voxels.values[outIndex] = voxel;

}

[[stage(compute), workgroup_size(128)]]
fn main_createLUT([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	doIgnore();

	var childIndex = GlobalInvocationID.x;

	var maxChildChunks = uniforms.chunkGridSize * uniforms.chunkGridSize * uniforms.chunkGridSize;
	if(childIndex >= maxChildChunks){
		return;
	}

	var numTriangles = atomicLoad(&grids.values[uniforms.tricountGridOffset + childIndex]);

	var offset = 0u;
	if(numTriangles > 0u){
		offset = atomicAdd(&chunks.values[uniforms.chunkIndex].lutCounter, numTriangles);
	}

	var a10 = atomicExchange(&grids.values[uniforms.LutGridOffset + childIndex], offset);

}

[[stage(compute), workgroup_size(128)]]
fn main_sortTriangles([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

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

	var voxelPos = toVoxelPos(uniforms.chunkGridSize, center);
	var voxelIndex = toIndex1D(uniforms.chunkGridSize, voxelPos);

	var triangleOffset = atomicAdd(&grids.values[uniforms.LutGridOffset + voxelIndex], 1u);

	sortedIndices.values[3u * triangleOffset + 0u] = i0;
	sortedIndices.values[3u * triangleOffset + 1u] = i1;
	sortedIndices.values[3u * triangleOffset + 2u] = i2;

	colors.values[i0] = 1234u * voxelIndex;
	colors.values[i1] = 1234u * voxelIndex;
	colors.values[i2] = 1234u * voxelIndex;

}

`;


let tricountGridCellCount = chunkGridSize ** 3;
let lutGridCellCount = chunkGridSize ** 3;
let voxelGridCellCount = voxelGridSize ** 3;
let tricountGridOffset = 0;
let lutGridOffset = tricountGridCellCount;
let voxelGridOffset = lutGridOffset + lutGridCellCount;

let gpu_grids = null;
let gpu_metadata = null;
let gpu_chunks = null;
let gpu_voxels = null;


function initialize(renderer, node){

	let {device} = renderer;

	let tricountGridSize = 4 * chunkGridSize ** 3;
	let voxelGridByteSize = 16 * voxelGridSize ** 3;

	let allGridsSize = 2 * tricountGridSize + voxelGridByteSize;
	gpu_grids = device.createBuffer({size: allGridsSize, usage: storage_flags});

	gpu_metadata = device.createBuffer({size: 256, usage: storage_flags});

	gpu_chunks = device.createBuffer({size: 32 * 10_000, usage: storage_flags});
	gpu_voxels = device.createBuffer({size: 128_000_000, usage: storage_flags});

	// tricountGridOffset = 0;
	// lutGridOffset = chunkGridSize ** 3;
	// voxelGridOffset = lutGridOffset + chunkGridSize ** 3;

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
		view.setUint32(12, voxelGridSize, true);

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

		view.setUint32(80, tricountGridOffset, true);
		view.setUint32(84, lutGridOffset, true);
		view.setUint32(88, voxelGridOffset, true);

		let chunkIndex = 0;
		let voxelBaseIndex = 0;
		view.setUint32(92, chunkIndex, true);
		view.setUint32(96, voxelBaseIndex, true);

		device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
	}

	let gpu_sortedIndices = device.createBuffer({size: 4 * chunk.triangles.numIndices, usage: storage_flags});
	chunk.triangles.gpu_sortedIndices = gpu_sortedIndices;

	let bindGroups = [
		{
			location: 0,
			entries: [
				{binding:  0, resource: {buffer: uniformBuffer}},
				{binding: 10, resource: {buffer: chunk.triangles.gpu_indices}},
				{binding: 11, resource: {buffer: chunk.triangles.gpu_position}},
				{binding: 12, resource: {buffer: chunk.triangles.gpu_color}},

				{binding: 20, resource: {buffer: gpu_grids}},
				{binding: 30, resource: {buffer: gpu_chunks}},
				{binding: 31, resource: {buffer: gpu_voxels}},
				{binding: 32, resource: {buffer: gpu_sortedIndices}},
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

	renderer.runCompute({
		code: csVoxelizing,
		entryPoint: "main_gatherVoxels",
		bindGroups: bindGroups,
		dispatchGroups: [Math.ceil((voxelGridSize ** 3) / 128)],
	});

	renderer.runCompute({
		code: csVoxelizing,
		entryPoint: "main_createLUT",
		bindGroups: bindGroups,
		dispatchGroups: [Math.ceil((chunkGridSize ** 3) / 128)],
	});


	renderer.runCompute({
		code: csVoxelizing,
		entryPoint: "main_sortTriangles",
		bindGroups: bindGroups,
		dispatchGroups: [Math.ceil(numTriangles / 128)],
	});



	let numVoxels = new Uint32Array(await renderer.readBuffer(gpu_chunks, 0, 4))[0];

	let pLutGrid = renderer.readBuffer(gpu_grids, 4 * lutGridOffset, 4 * lutGridCellCount);
	let pTricountGrid = renderer.readBuffer(gpu_grids, 4 * tricountGridOffset, 4 * tricountGridCellCount);
	let lutGrid = new Uint32Array(await pLutGrid);
	let tricountGrid = new Uint32Array(await pTricountGrid);

	for(let i = 0; i < 8; i++){
		let numTriangles = tricountGrid[i]
		let triangleOffset = lutGrid[i] - numTriangles;

		if(numTriangles === 0){
			chunk.children[i] = null;
		}else{

			let child = new Chunk();

			child.boundingBox = computeChildBox(chunk.boundingBox, i);
			child.triangles = {
				gpu_position   : chunk.triangles.gpu_position,
				gpu_uv         : chunk.triangles.gpu_uv,
				gpu_color      : chunk.triangles.gpu_color,
				gpu_indices    : gpu_sortedIndices,
				firstIndex     : 3 * triangleOffset,
				numIndices     : 3 * numTriangles,
			};
			child.voxels = {
				gpu_voxels     : chunk.gpu_voxels,
				firstVoxel     : 0,
				numVoxels      : 0,
			};

			chunk.children[i] = child;
		}
		
	}

	{

		let positions = node.geometry.findBuffer("position");
		let colors = node.geometry.findBuffer("color");
		let uvs = node.geometry.findBuffer("uv");
		// let indices = node.geometry.indices;

		let selected = chunk.children[0];
		let firstIndex = selected.triangles.firstIndex;
		let numIndices = selected.triangles.numIndices;
		let indexBuffer = await renderer.readBuffer(
			selected.triangles.gpu_indices, 
			4 * 3 * firstIndex, 
			4 * 3 * numIndices);
		let indices = new Uint32Array(indexBuffer);


		potree.onUpdate( () => {
			potree.renderer.drawMesh({
				positions, 
				colors, 
				uvs,
				indices,
				image: node.material.image
			});
		});
		
	}

	potree.onUpdate( () => {
		
		let selected = chunk.children[0];
		let position = selected.boundingBox.center();
		let scale = new Vector3(1, 1, 1).multiplyScalar(selected.boundingBox.size().x);
		let color = new Vector3(255, 0, 0);

		potree.renderer.drawBoundingBox(position, scale, color);

	});

	// potree.renderer.onDraw( (drawstate) => {

	// 	let cubeSize = chunk.boundingBox.max.x - chunk.boundingBox.min.x;
	// 	let voxelSize = cubeSize / voxelGridSize;
	// 	let voxels = {gpu_chunks, gpu_voxels, numVoxels, voxelSize};

	// 	renderVoxels(drawstate, voxels);
	// });



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