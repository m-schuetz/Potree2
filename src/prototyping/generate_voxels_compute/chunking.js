
import { Vector3 } from "../../math/Vector3.js";
import {Chunk, voxelGridSize, toIndex1D, toIndex3D, computeChildBox} from "./common.js";
import {storage_flags, uniform_flags} from "./common.js";
import { transferTriangles } from "./transferTriangles.js";
import {Geometry, Mesh, Box3} from "potree";

export let csChunking = `

[[block]] struct Uniforms {
	chunkGridSize      : u32;
	pad_1              : u32;
	pad_0              : u32;
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

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), group(0)]] var mySampler: sampler;
[[binding(2), group(0)]] var myTexture: texture_2d<f32>;

[[binding(10), group(0)]] var<storage, read_write> indices   : U32s;
[[binding(11), group(0)]] var<storage, read_write> positions : F32s;
[[binding(12), group(0)]] var<storage, read_write> colors    : U32s;
[[binding(13), group(0)]] var<storage, read_write> uvs       : F32s;

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

fn getNumChunkGridCells() -> u32 {
	var chunkGridSize = uniforms.chunkGridSize;
	var numChunkGridCells = chunkGridSize * chunkGridSize * chunkGridSize;

	return numChunkGridCells;
};

fn getTricountGridOffset() -> u32 {
	return 0u;
};

fn getLutGridOffset() -> u32 {
	return getNumChunkGridCells();
};

fn getSortedCounterGridOffset() -> u32 {
	return 2u * getNumChunkGridCells();
};

fn getBatchGridOffset() -> u32 {
	var numChunkGridCells = getNumChunkGridCells();
	var numBatchGridElements = 3u * numChunkGridCells;
	var batchGridOffset = uniforms.batchIndex * numBatchGridElements;

	return batchGridOffset;
};

fn doIgnore(){
	
	_ = uniforms;

	_ = &indices;
	_ = &positions;
	_ = &colors;
	_ = &uvs;
	_ = &grids;
	_ = &batches;
	_ = &sortedTriangles;
	_ = mySampler;
	_ = myTexture;
}

[[stage(compute), workgroup_size(128)]]
fn main_accumulate([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	doIgnore();

	let batch = &batches.values[uniforms.batchIndex];

	if(GlobalInvocationID.x >= (*batch).numTriangles){
		return;
	}

	// for batch-wise buffer binding
	var triangleIndex = GlobalInvocationID.x;
	// for full buffer binding
	var firstTriangle = (*batch).firstTriangle;
	// var triangleIndex = (*batch).firstTriangle + GlobalInvocationID.x;

	var i0 = indices.values[3u * triangleIndex + 0u] - 3u * firstTriangle;
	var i1 = indices.values[3u * triangleIndex + 1u] - 3u * firstTriangle;
	var i2 = indices.values[3u * triangleIndex + 2u] - 3u * firstTriangle;

	var p0 = loadPosition(i0);
	var p1 = loadPosition(i1);
	var p2 = loadPosition(i2);
	var center = (p0 + p1 + p2) / 3.0;

	var chunkPos = toCellPos(uniforms.chunkGridSize, center);
	var chunkIndex = toIndex1D(uniforms.chunkGridSize, chunkPos);

	var batchGridOffset = getBatchGridOffset();

	var a00 = atomicAdd(&grids.values[batchGridOffset + getTricountGridOffset() + chunkIndex], 1u);

}

[[stage(compute), workgroup_size(128)]]
fn main_create_lut([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	doIgnore();

	var childIndex = GlobalInvocationID.x;

	var maxChildChunks = uniforms.chunkGridSize * uniforms.chunkGridSize * uniforms.chunkGridSize;
	if(childIndex >= maxChildChunks){
		return;
	}

	var batchGridOffset = getBatchGridOffset();

	var numTriangles = atomicLoad(&grids.values[batchGridOffset + getTricountGridOffset() + childIndex]);

	var offset = 0u;
	if(numTriangles > 0u){
		offset = atomicAdd(&batches.values[uniforms.batchIndex].lutCounter, numTriangles);
	}

	var a10 = atomicExchange(&grids.values[batchGridOffset + getLutGridOffset() + childIndex], offset);

}

fn mixColors(chunkIndex : u32, color : u32) -> u32 {

	var SPECTRAL : array<vec3<u32>, 6> = array<vec3<u32>, 6>(
		vec3<u32>(213u,  62u,  79u),
		vec3<u32>(252u, 141u,  89u),
		vec3<u32>(254u, 224u, 139u),
		vec3<u32>(230u, 245u, 152u),
		vec3<u32>(153u, 213u, 148u),
		vec3<u32>( 50u, 136u, 189u),
	);

	var gradIndex = (chunkIndex * 5u) % 6u;
	var cR = f32(SPECTRAL[gradIndex].x);
	var cG = f32(SPECTRAL[gradIndex].y);
	var cB = f32(SPECTRAL[gradIndex].z);

	var R = (color >>  0u) & 0xFFu;
	var G = (color >>  8u) & 0xFFu;
	var B = (color >> 16u) & 0xFFu;

	var w = 0.3;
	R = u32((1.0 - w) * f32(R) + w * cR);
	G = u32((1.0 - w) * f32(R) + w * cG);
	B = u32((1.0 - w) * f32(R) + w * cB);

	var newColor = 
		(R <<  0u) |
		(G <<  8u) |
		(B << 16u);

	return newColor;
};

[[stage(compute), workgroup_size(128)]]
fn main_sort_triangles([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	doIgnore();

	let batch = &batches.values[uniforms.batchIndex];
	var numTriangles = (*batch).numTriangles;
	var firstTriangle = (*batch).firstTriangle;

	if(GlobalInvocationID.x >= numTriangles){
		return;
	}

	// var triangleIndex = firstTriangle + GlobalInvocationID.x;
	var triangleIndex = GlobalInvocationID.x;

	var i0 = indices.values[3u * triangleIndex + 0u] - 3u * firstTriangle;
	var i1 = indices.values[3u * triangleIndex + 1u] - 3u * firstTriangle;
	var i2 = indices.values[3u * triangleIndex + 2u] - 3u * firstTriangle;

	var p0 = loadPosition(i0);
	var p1 = loadPosition(i1);
	var p2 = loadPosition(i2);
	var center = (p0 + p1 + p2) / 3.0;

	var chunkPos = toCellPos(uniforms.chunkGridSize, center);
	var chunkIndex = toIndex1D(uniforms.chunkGridSize, chunkPos);

	var batchGridOffset = getBatchGridOffset();

	// var triangleOffset = atomicAdd(&grids.values[batchGridOffset + getLutGridOffset() + chunkIndex], 1u);
	var lutOffset = atomicLoad(&grids.values[batchGridOffset + getLutGridOffset() + chunkIndex]);
	var triangleOffset = lutOffset + atomicAdd(&grids.values[batchGridOffset + getSortedCounterGridOffset() + chunkIndex], 1u);

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
	// sortedTriangles.values[offset_color + 3u * triangleOffset + 0u] = mixColors(chunkIndex, colors.values[i0]);
	// sortedTriangles.values[offset_color + 3u * triangleOffset + 1u] = mixColors(chunkIndex, colors.values[i1]);
	// sortedTriangles.values[offset_color + 3u * triangleOffset + 2u] = mixColors(chunkIndex, colors.values[i2]);

	// var color = 12345u * (chunkIndex + 1u);
	// sortedTriangles.values[offset_color + 3u * triangleOffset + 0u] = color;
	// sortedTriangles.values[offset_color + 3u * triangleOffset + 1u] = color;
	// sortedTriangles.values[offset_color + 3u * triangleOffset + 2u] = color;
}

`;

const maxBatchSize = 1_000_000;
const chunkGridSize = 3;
const chunkGridCellCount = chunkGridSize ** 3;

async function process(renderer, node, chunk){

	console.log("chunking start", (performance.now() / 1000).toFixed(3));

	let {device} = renderer;

	let cube = node.boundingBox.cube();
	let buffers = {
		position     : node.geometry.findBuffer("position"),
		uv           : node.geometry.findBuffer("uv"),
		color        : node.geometry.findBuffer("color"),
		indices      : node.geometry.indices,
	};
	let gpu_indices   = renderer.getGpuBuffer(buffers.indices);
	let gpu_positions = renderer.getGpuBuffer(buffers.position);
	let gpu_colors    = renderer.getGpuBuffer(buffers.color);
	let gpu_uvs       = renderer.getGpuBuffer(buffers.uv);

	let numTriangles = buffers.indices.length / 3;
	let numBatches = Math.ceil(numTriangles / maxBatchSize);

	let gpu_grids = device.createBuffer({
		size: numBatches * (3 * 4 * chunkGridCellCount), 
		usage: storage_flags
	});

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

		// INIT UNIFORMS
		let uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});
		{
			let buffer = new ArrayBuffer(256);
			let view = new DataView(buffer);

			view.setUint32 (  0, chunkGridSize, true);
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

		let batchIndexByteOffset         = 4 * 3 * batch.firstTriangle;
		let batchIndexByteSize           = 4 * 3 * batch.numTriangles;
		let batchPositionsByteOffset     = 4 * 9 * batch.firstTriangle;
		let batchPositionsByteSize       = 4 * 9 * batch.numTriangles;
		let batchColorsByteOffset        = 4 * 3 * batch.firstTriangle;
		let batchColorsByteSize          = 4 * 3 * batch.numTriangles;
		let batchUVsByteOffset           = 4 * 6 * batch.firstTriangle;
		let batchUVsByteSize             = 4 * 6 * batch.numTriangles;


		let bindGroups = [{
			location: 0,
			entries: [
				{binding:  0, resource: {buffer: uniformBuffer}},
				{binding:  1, resource: renderer.getDefaultSampler()},
				{binding:  2, resource: renderer.getGpuTexture(node.material.image)},

				{binding: 10, resource: {buffer: gpu_indices, offset: batchIndexByteOffset, size: batchIndexByteSize}},
				{binding: 11, resource: {buffer: gpu_positions, offset: batchPositionsByteOffset, size: batchPositionsByteSize}},
				{binding: 12, resource: {buffer: gpu_colors, offset: batchColorsByteOffset, size: batchColorsByteSize}},
				{binding: 13, resource: {buffer: gpu_uvs, offset: batchUVsByteOffset, size: batchUVsByteSize}},
				// {binding: 10, resource: {buffer: gpu_indices}},
				// {binding: 11, resource: {buffer: gpu_positions}},
				// {binding: 12, resource: {buffer: gpu_colors}},

				{binding: 20, resource: {buffer: gpu_grids}},
				{binding: 30, resource: {buffer: gpu_batches}},
				{binding: 50, resource: {buffer: gpu_sortedTriangles}},
			],
		}];

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

	}


	// PASS 2: compute chunks
	// - compute occupied chunks
	// - each chunk may have triangles in any of the batches.
	let chunks = [];
	{
		let gridBuffers = await potree.renderer.readBuffer(gpu_grids, 0, numBatches * (3 * 4 * chunkGridCellCount));
		let view = new DataView(gridBuffers);
		for(let i = 0; i < chunkGridCellCount; i++){
			let chunk = {
				numTriangles: 0,
				fragments: [],
			};
			chunks[i] = chunk;
		}

		for(let batchIndex = 0; batchIndex < batches.length; batchIndex++){
			let batch = batches[batchIndex];

			let numTrianglesInBatch = 0;
			for(let chunkIndex = 0; chunkIndex < chunkGridCellCount; chunkIndex++){
				let batchGridOffset = 4 * batchIndex * 3 * chunkGridCellCount;
				let numTrianglesInChunk = view.getUint32(batchGridOffset + 4 * chunkIndex, true);
				let firstTriangleInChunk = view.getUint32(batchGridOffset + 4 * chunkGridCellCount + 4 * chunkIndex, true);

				if(numTrianglesInChunk > 0){
					chunks[chunkIndex].numTriangles += numTrianglesInChunk;

					let fragment = {
						batchIndex: batchIndex, 
						numTriangles: numTrianglesInChunk, 
						firstTriangle: firstTriangleInChunk,
					}
					chunks[chunkIndex].fragments.push(fragment);
				}

				numTrianglesInBatch += numTrianglesInChunk;
			}
		}
	}

	// PASS 3: Copy triangles from all batches to chunks
	{
		for(let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++){
			let chunk = chunks[chunkIndex];

			if(chunk.numTriangles === 0){
				continue;
			}

			let gpu_positions = device.createBuffer({size: 12 * 3 * chunk.numTriangles, usage: storage_flags});
			let gpu_colors = device.createBuffer({size: 4 * 3 * chunk.numTriangles, usage: storage_flags});

			chunk.gpu_positions = gpu_positions;
			chunk.gpu_colors = gpu_colors;

			let numTransfered = 0;
			for(let fragment of chunk.fragments){
			// {
				// let fragment = chunk.fragments[3];
				
				let {batchIndex, numTriangles, firstTriangle} = fragment;
				let batch = batches[batchIndex];

				// copy triangles from batch to chunk
				transferTriangles(
					renderer, 
					batch, 
					chunk, 
					chunkGridSize, 
					numTriangles, 
					firstTriangle,
					numTransfered
				);

				numTransfered += numTriangles;
			}

		}
	}

	// PASS 4: Finalize by turning the chunks into scene nodes
	let nodes = [];
	for(let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++){
		let chunk = chunks[chunkIndex];

		if(chunk.numTriangles === 0){
			continue;
		}

		let bPositions = await potree.renderer.readBuffer(chunk.gpu_positions, 0, 4 * 9 * chunk.numTriangles);
		let bColors = await potree.renderer.readBuffer(chunk.gpu_colors, 0, 4 * 3 * chunk.numTriangles);
		let positions = new Float32Array(bPositions);
		let colors = new Uint32Array(bColors);
		let indices = new Uint32Array(3 * chunk.numTriangles);
		for(let i = 0; i < 3 * chunk.numTriangles; i++){
			indices[i] = i;
		}

		let geometry = new Geometry();
		geometry.buffers = [
			{name: "position", buffer: positions},
			{name: "color", buffer: colors},
		];
		geometry.indices = indices;

		let mesh = new Mesh(`chunk_${chunkIndex}`, geometry);
		// TODO BOUNDING BOX
		{
			let chunkCoord = toIndex3D(chunkGridSize, chunkIndex);
			let cubeSize = cube.max.x - cube.min.x;
			let chunkSize = cube.size().divideScalar(chunkGridSize);
			let chunkMin = new Vector3(
				cubeSize * chunkCoord.x / chunkGridSize + cube.min.x,
				cubeSize * chunkCoord.y / chunkGridSize + cube.min.y,
				cubeSize * chunkCoord.z / chunkGridSize + cube.min.z,
			);
			let chunkMax = chunkMin.clone().add(chunkSize);
			
			mesh.boundingBox = new Box3(chunkMin, chunkMax);

		}

		nodes.push(mesh);
	}

	console.log("chunking end", (performance.now() / 1000).toFixed(3));

	return nodes;
}

export async function doChunking(renderer, node){

	let nodes = await process(renderer, node);
	
	return nodes;
}