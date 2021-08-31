
import { Vector3 } from "../../math/Vector3.js";
import {Chunk, voxelGridSize, toIndex1D, toIndex3D, computeChildBox} from "./common.js";
import {storage_flags, uniform_flags} from "./common.js";
import {SPECTRAL} from "potree";

export let csChunking = `

[[block]] struct Uniforms {
	chunkGridSize      : u32;
	tricountGridOffset : u32;
	lutGridOffset      : u32;
	batchIndex         : u32;
	boxMin             : vec4<f32>;      // offset(16)
	boxMax             : vec4<f32>;      // offset(32)
};

struct Batch {
	numTriangles      : u32;
	firstTriangle     : u32;
	lutCounter        : atomic<u32>;
};

[[block]] struct F32s { values : [[stride(4)]] array<f32>; };
[[block]] struct U32s { values : [[stride(4)]] array<u32>; };
[[block]] struct I32s { values : [[stride(4)]] array<i32>; };
[[block]] struct AU32s { values : [[stride(4)]] array<atomic<u32>>; };
[[block]] struct AI32s { values : [[stride(4)]] array<atomic<i32>>; };
[[block]] struct Batches { values : [[stride(32)]] array<Batch>; };

[[binding( 0), group(0)]] var<uniform> uniforms : Uniforms;

[[binding(10), group(0)]] var<storage, read_write> indices   : U32s;
[[binding(11), group(0)]] var<storage, read_write> positions : F32s;
[[binding(12), group(0)]] var<storage, read_write> colors    : U32s;

[[binding(20), group(0)]] var<storage, read_write> grids     : AU32s;

[[binding(30), group(0)]] var<storage, read_write> batches   : Batches;

[[binding(50), group(0)]] var<storage, read_write> sortedTriangles : U32s;

fn toCellPos(gridSize : u32, position : vec3<f32>) -> vec3<f32>{

	var bbMin = vec3<f32>(uniforms.boxMin.x, uniforms.boxMin.y, uniforms.boxMin.z);
	var bbMax = vec3<f32>(uniforms.boxMax.x, uniforms.boxMax.y, uniforms.boxMax.z);
	var bbSize = bbMax - bbMin;
	var cubeSize = max(max(bbSize.x, bbSize.y), bbSize.z);
	var fGridSize = f32(gridSize);

	var gx = fGridSize * (position.x - uniforms.boxMin.x) / cubeSize;
	var gy = fGridSize * (position.y - uniforms.boxMin.y) / cubeSize;
	var gz = fGridSize * (position.z - uniforms.boxMin.z) / cubeSize;

	return vec3<f32>(gx, gy, gz);
}

fn toIndex1D(gridSize : u32, cellPos : vec3<f32>) -> u32{

	var icoord = vec3<u32>(cellPos);

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

	ignore(indices);
	ignore(positions);
	ignore(colors);
	ignore(grids);
	ignore(batches);
	ignore(sortedTriangles);
}

[[stage(compute), workgroup_size(128)]]
fn main_accumulate([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	doIgnore();

	let batch = &batches.values[uniforms.batchIndex];

	if(GlobalInvocationID.x >= (*batch).numTriangles){
		return;
	}

	var triangleIndex = (*batch).firstTriangle + GlobalInvocationID.x;

	var i0 = indices.values[3u * triangleIndex + 0u];
	var i1 = indices.values[3u * triangleIndex + 1u];
	var i2 = indices.values[3u * triangleIndex + 2u];

	var p0 = loadPosition(i0);
	var p1 = loadPosition(i1);
	var p2 = loadPosition(i2);
	var center = (p0 + p1 + p2) / 3.0;

	var chunkPos = toCellPos(uniforms.chunkGridSize, center);
	var chunkIndex = toIndex1D(uniforms.chunkGridSize, chunkPos);

	var a00 = atomicAdd(&grids.values[uniforms.tricountGridOffset + chunkIndex], 1u);

}

[[stage(compute), workgroup_size(128)]]
fn main_create_lut([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	doIgnore();

	var childIndex = GlobalInvocationID.x;

	var maxChildChunks = uniforms.chunkGridSize * uniforms.chunkGridSize * uniforms.chunkGridSize;
	if(childIndex >= maxChildChunks){
		return;
	}

	var numTriangles = atomicLoad(&grids.values[uniforms.tricountGridOffset + childIndex]);

	var offset = 0u;
	if(numTriangles > 0u){
		offset = atomicAdd(&batches.values[uniforms.batchIndex].lutCounter, numTriangles);
	}

	var a10 = atomicExchange(&grids.values[uniforms.lutGridOffset + childIndex], offset);

}

[[stage(compute), workgroup_size(128)]]
fn main_sort_triangles([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	doIgnore();

	let batch = &batches.values[uniforms.batchIndex];
	var numTriangles = (*batch).numTriangles;
	var firstTriangle = (*batch).firstTriangle;

	if(GlobalInvocationID.x >= numTriangles){
		return;
	}

	var triangleIndex = firstTriangle + GlobalInvocationID.x;

	var i0 = indices.values[3u * triangleIndex + 0u];
	var i1 = indices.values[3u * triangleIndex + 1u];
	var i2 = indices.values[3u * triangleIndex + 2u];

	var p0 = loadPosition(i0);
	var p1 = loadPosition(i1);
	var p2 = loadPosition(i2);
	var center = (p0 + p1 + p2) / 3.0;

	var chunkPos = toCellPos(uniforms.chunkGridSize, center);
	var chunkIndex = toIndex1D(uniforms.chunkGridSize, chunkPos);

	var triangleOffset = atomicAdd(&grids.values[uniforms.lutGridOffset + chunkIndex], 1u);

	// 3 vertices, 9 float values per triangle
	var offset_pos = 0u;
	var offset_color = 9u * numTriangles;

	var X0 = bitcast<u32>(p0.x);
	var Y0 = bitcast<u32>(p0.y);
	var Z0 = bitcast<u32>(p0.z);
	var X1 = bitcast<u32>(p1.x);
	var Y1 = bitcast<u32>(p1.y);
	var Z1 = bitcast<u32>(p1.z);
	var X2 = bitcast<u32>(p2.x);
	var Y2 = bitcast<u32>(p2.y);
	var Z2 = bitcast<u32>(p2.z);

	sortedTriangles.values[offset_pos + 9u * triangleOffset + 0u] = X0;
	sortedTriangles.values[offset_pos + 9u * triangleOffset + 1u] = Y0;
	sortedTriangles.values[offset_pos + 9u * triangleOffset + 2u] = Z0;
	sortedTriangles.values[offset_pos + 9u * triangleOffset + 3u] = X1;
	sortedTriangles.values[offset_pos + 9u * triangleOffset + 4u] = Y1;
	sortedTriangles.values[offset_pos + 9u * triangleOffset + 5u] = Z1;
	sortedTriangles.values[offset_pos + 9u * triangleOffset + 6u] = X2;
	sortedTriangles.values[offset_pos + 9u * triangleOffset + 7u] = Y2;
	sortedTriangles.values[offset_pos + 9u * triangleOffset + 8u] = Z2;

	

	sortedTriangles.values[offset_color + 3u * triangleOffset + 0u] = colors.values[i0];
	sortedTriangles.values[offset_color + 3u * triangleOffset + 1u] = colors.values[i1];
	sortedTriangles.values[offset_color + 3u * triangleOffset + 2u] = colors.values[i2];

	var color = 12345u * chunkIndex;
	sortedTriangles.values[offset_color + 3u * triangleOffset + 0u] = color;
	sortedTriangles.values[offset_color + 3u * triangleOffset + 1u] = color;
	sortedTriangles.values[offset_color + 3u * triangleOffset + 2u] = color;


}

`;

const maxBatchSize = 1_000_000;
const chunkGridSize = 8;
const tricountGridCellCount = chunkGridSize ** 3;
const lutGridCellCount = chunkGridSize ** 3;
const allGridsCellCount = tricountGridCellCount + lutGridCellCount;
const tricountGridOffset = 0;
const lutGridOffset = tricountGridCellCount;
const allGridsByteSize = 4 * allGridsCellCount;

let gpu_grids = null;
// let gpu_batches = null;
let gpu_empty = null;

const passes = {};

function initialize(renderer, node){

	let {device} = renderer;

	gpu_grids = device.createBuffer({size: allGridsByteSize, usage: storage_flags});
	// gpu_batches = device.createBuffer({size: 32 * 10_000, usage: storage_flags});
	gpu_empty = device.createBuffer({size: 32, usage: storage_flags});

	{ // PASS - ACCUMULATE
		let pipeline = renderer.createComputePipeline({code: csChunking, entryPoint: "main_accumulate"});

		passes["accumulate"] = {pipeline};
	}

}

async function process(renderer, node, chunk){

	let {device} = renderer;

	let cube = node.boundingBox.cube();
	let buffers = {
		position     : node.geometry.findBuffer("position"),
		uv           : node.geometry.findBuffer("uv"),
		color        : node.geometry.findBuffer("color"),
		indices      : node.geometry.indices,
	};
	let gpu_indices   = renderer.getGpuBuffer(buffers.indices);
	let gpu_uvs       = renderer.getGpuBuffer(buffers.uv);
	let gpu_positions = renderer.getGpuBuffer(buffers.position);
	let gpu_colors    = renderer.getGpuBuffer(buffers.color);

	let numTriangles = buffers.indices.length / 3;
	let numBatches = Math.ceil(numTriangles / maxBatchSize);

	// INIT BATCH DATA
	let batches = [];
	let gpu_batches = device.createBuffer({size: 32 * numBatches, usage: storage_flags});
	{
		let batchesData = new ArrayBuffer(32 * numBatches);
		let stride = 32;
		let view = new DataView(batchesData);

		for(let i = 0; i < numBatches; i++){
			let batchFirstTriangle = i * maxBatchSize;
			let batchNumTriangles = Math.min(numTriangles - batchFirstTriangle, maxBatchSize);
			let batch = {
				firstTriangle : batchFirstTriangle,
				numTriangles  : batchNumTriangles,
			};

			view.setUint32(stride * i + 0, batch.numTriangles, true);
			view.setUint32(stride * i + 4, batch.firstTriangle, true);

			batches.push(batch);
		}

		device.queue.writeBuffer(gpu_batches, 0, batchesData, 0, batchesData.byteLength);
	}

	// PASS 1: Sort triangles into buckets batch-wise
	for(let batchIndex = 0; batchIndex < numBatches; batchIndex++){
		let batch = batches[batchIndex];

		console.log(batch);

		// INIT UNIFORMS
		let uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});
		{
			let buffer = new ArrayBuffer(256);
			let view = new DataView(buffer);

			view.setUint32 (  0, chunkGridSize, true);
			view.setUint32 (  4, tricountGridOffset, true);
			view.setUint32 (  8, lutGridOffset, true);
			view.setUint32 ( 12, batchIndex, true);
			view.setFloat32( 16, cube.min.x, true);
			view.setFloat32( 20, cube.min.y, true);
			view.setFloat32( 24, cube.min.z, true);
			view.setFloat32( 32, cube.max.z, true);
			view.setFloat32( 36, cube.max.y, true);
			view.setFloat32( 40, cube.max.z, true);

			device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
		}

		// 3 vertices; 12 bytes XYZ per vertex; 4 bytes RGB per vertex
		let sortedTrianglesByteSize = 3 * 16 * batch.numTriangles;
		let gpu_sortedTriangles = device.createBuffer({size: sortedTrianglesByteSize, usage: storage_flags});

		let bindGroups = [{
			location: 0,
			entries: [
				{binding:  0, resource: {buffer: uniformBuffer}},
				{binding: 10, resource: {buffer: gpu_indices}},
				{binding: 11, resource: {buffer: gpu_positions}},
				{binding: 12, resource: {buffer: gpu_colors}},

				{binding: 20, resource: {buffer: gpu_grids}},
				{binding: 30, resource: {buffer: gpu_batches}},
				{binding: 50, resource: {buffer: gpu_sortedTriangles}},
			],
		}];

		// reset counters and LUT
		renderer.fillBuffer(gpu_grids, 0, (tricountGridCellCount + lutGridCellCount));

		// accumulate
		renderer.runCompute({
			code: csChunking,
			entryPoint: "main_accumulate",
			bindGroups: bindGroups,
			dispatchGroups: [Math.ceil(batch.numTriangles / 128)],
		});

		// create LUT
		renderer.runCompute({
			code: csChunking,
			entryPoint: "main_create_lut",
			bindGroups: bindGroups,
			dispatchGroups: [Math.ceil((chunkGridSize ** 3) / 128)],
		});

		// partition triangles
		renderer.runCompute({
			code: csChunking,
			entryPoint: "main_sort_triangles",
			bindGroups: bindGroups,
			dispatchGroups: [Math.ceil(batch.numTriangles / 128)],
		});

		batch.gpu_sortedTriangles = gpu_sortedTriangles;

		renderer.readBuffer(gpu_sortedTriangles, 0, 3 * 16 * batch.numTriangles).then(result => {
			let positions = new Float32Array(result, 0, 9 * batch.numTriangles);
			let colors = new Uint32Array(result, 4 * 9 * batch.numTriangles, 3 * batch.numTriangles);

			let mesh = {
				positions, 
				colors,
				image: node.material.image,
			};

			batch.mesh = mesh;

			
		});

		// break;
	}

	potree.onUpdate( () => {
		for(let batch of batches){
			if(batch.mesh){
				potree.renderer.drawMesh(batch.mesh);
			}
		}
	});

	

	// PASS 2: create chunks
	// - compute occupied chunks
	// - each chunk may have triangles in any of the batches.
	//   Figure out how many triangles, then run through 
	//   batches and copy them to a single buffer per chunk


	// PASS 3: Finalize by turning the chunks into scene nodes


}

export async function doChunking(renderer, node){

	initialize(renderer, node);

	await process(renderer, node);
	
}