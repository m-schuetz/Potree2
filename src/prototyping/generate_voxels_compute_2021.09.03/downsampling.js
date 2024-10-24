
import {Chunk, voxelGridSize, chunkGridSize, toIndex1D, toIndex3D, computeChildBox} from "./common.js";
import {storage_flags, uniform_flags, maxTrianglesPerNode} from "./common.js";
import {renderVoxelsLOD} from "./renderVoxelsLOD.js";

export let csVoxelizing = `

struct Uniforms {
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

struct F32s { values : array<f32>; };
struct U32s { values : array<u32> };
struct I32s { values : array<i32>; };
struct AU32s { values : array<atomic<u32>>; };
struct AI32s { values : array<atomic<i32>>; };
struct Chunks { values : array<Chunk>; };
struct Voxels { values : array<Voxel> };

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

	if(GlobalInvocationID.x >= uniforms.numTriangles){
		return;
	}

	var triangleIndex = uniforms.firstTriangle + GlobalInvocationID.x;

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


	if(GlobalInvocationID.x >= uniforms.numTriangles){
		return;
	}

	var triangleIndex = uniforms.firstTriangle + GlobalInvocationID.x;

	var i0 = indices.values[3u * triangleIndex + 0u];
	var i1 = indices.values[3u * triangleIndex + 1u];
	var i2 = indices.values[3u * triangleIndex + 2u];

	var p0 = loadPosition(i0);
	var p1 = loadPosition(i1);
	var p2 = loadPosition(i2);
	var center = (p0 + p1 + p2) / 3.0;

	var chunkCoord = toVoxelPos(uniforms.chunkGridSize, center);
	var chunkIndex = toIndex1D(uniforms.chunkGridSize, chunkCoord);

	var triangleOffset = atomicAdd(&grids.values[uniforms.LutGridOffset + chunkIndex], 1u);

	sortedIndices.values[3u * triangleOffset + 0u] = i0;
	sortedIndices.values[3u * triangleOffset + 1u] = i1;
	sortedIndices.values[3u * triangleOffset + 2u] = i2;

	// colors.values[i0] = (255u * chunkIndex) / 7u;
	// colors.values[i1] = (255u * chunkIndex) / 7u;
	// colors.values[i2] = (255u * chunkIndex) / 7u;

}

`;

let tricountGridCellCount = chunkGridSize ** 3;
let lutGridCellCount = chunkGridSize ** 3;
let voxelGridCellCount = voxelGridSize ** 3;
let tricountGridOffset = 0;
let lutGridOffset = tricountGridCellCount;
let voxelGridOffset = lutGridOffset + lutGridCellCount;
let tricountGridSize = 4 * tricountGridCellCount;
let voxelGridByteSize = 16 * voxelGridCellCount;
let allGridsSize = 2 * tricountGridSize + voxelGridByteSize;

let stateCache = new Map();
function getState(renderer, node){

	if(stateCache.has(node)){
		return stateCache.get(node);
	}else{
		let {device} = renderer;

		let currentlyProcessing = null;
		let gpu_grids = device.createBuffer({size: allGridsSize, usage: storage_flags});
		let gpu_chunks = device.createBuffer({size: 32 * 10_000, usage: storage_flags});
		let gpu_voxels = device.createBuffer({size: 128_000_000, usage: storage_flags});

		let passes = {};

		{ // PASS - ACCUMULATE
			let pipeline = renderer.createComputePipeline({code: csVoxelizing, entryPoint: "main_accumulate"});

			passes["accumulate"] = {pipeline};
		}

		{ // PASS - GATHER VOXELS
			let pipeline = renderer.createComputePipeline({code: csVoxelizing, entryPoint: "main_gatherVoxels"});

			passes["gather_voxels"] = {pipeline};
		}

		{ // PASS - CREATE LUT
			let pipeline = renderer.createComputePipeline({code: csVoxelizing, entryPoint: "main_createLUT"});

			passes["create_lut"] = {pipeline};
		}

		{ // PASS - SORT TRIANGLES
			let pipeline = renderer.createComputePipeline({code: csVoxelizing, entryPoint: "main_sortTriangles"});

			passes["sort_triangles"] = {pipeline};
		}

		let numChunksProcessed = 0;
		let numVoxelsProcessed = 0;
		let chunkList = [];

		let state = {
			currentlyProcessing, gpu_grids, gpu_chunks, gpu_voxels,
			passes, numChunksProcessed, numVoxelsProcessed, chunkList,
		};

		stateCache.set(node, state);

		return state;
	}

}


async function processChunk(renderer, node, chunk){

	console.log("processChunk start", (performance.now() / 1000).toFixed(3));

	let state = getState(renderer, node);

	if(state.currentlyProcessing){
		return;
	}else{
		state.currentlyProcessing = chunk;
	}

	// console.log(`=======================================================================`);
	// console.log(`processing ${chunk.id}, numChunksProcessed: ${numChunksProcessed}`);
	// console.log(`=======================================================================`);

	let {device} = renderer;

	chunk.index = state.numChunksProcessed;
	chunk.voxels.firstVoxel = state.numVoxelsProcessed;

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

		
		view.setUint32(92, chunk.index, true);
		view.setUint32(96, state.numVoxelsProcessed, true);

		device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
	}

	let gpu_sortedIndices = device.createBuffer({
		size: 4 * chunk.triangles.numIndices, 
		usage: storage_flags});
	chunk.triangles.gpu_sortedIndices = gpu_sortedIndices;

	renderer.fillBuffer(state.gpu_grids, 0, tricountGridCellCount + lutGridCellCount + 4 * voxelGridCellCount);

	let bindGroups = [{
		location: 0,
		entries: [
			{binding:  0, resource: {buffer: uniformBuffer}},
			{binding: 10, resource: {buffer: chunk.triangles.gpu_indices}},
			{binding: 11, resource: {buffer: chunk.triangles.gpu_position}},
			{binding: 12, resource: {buffer: chunk.triangles.gpu_color}},

			{binding: 20, resource: {buffer: state.gpu_grids}},
			{binding: 30, resource: {buffer: state.gpu_chunks}},
			{binding: 31, resource: {buffer: state.gpu_voxels}},
			{binding: 32, resource: {buffer: gpu_sortedIndices}},
		],
	}];

	let numTriangles = chunk.triangles.numIndices / 3;

	{
		let passes = state.passes;

		const commandEncoder = device.createCommandEncoder();
		const passEncoder = commandEncoder.beginComputePass();

		for(let bindGroupItem of bindGroups){

			let bindGroup = device.createBindGroup({
				layout: passes["accumulate"].pipeline.getBindGroupLayout(bindGroupItem.location),
				entries: bindGroupItem.entries,
			});

			passEncoder.setBindGroup(bindGroupItem.location, bindGroup);
		}

		{ // ACCUMULATE
			let {pipeline} = passes["accumulate"];

			passEncoder.setPipeline(pipeline);

			let dispatchGroups = [Math.ceil(numTriangles / 128)];
			passEncoder.dispatch(...dispatchGroups);
		}

		{ // GATHER VOXELS
			let {pipeline} = passes["gather_voxels"];

			passEncoder.setPipeline(pipeline);

			let dispatchGroups = [Math.ceil((voxelGridSize ** 3) / 128)];
			passEncoder.dispatch(...dispatchGroups);
		}

		{ // CREATE LUT
			let {pipeline} = passes["create_lut"];
			passEncoder.setPipeline(pipeline);

			let dispatchGroups = [Math.ceil((chunkGridSize ** 3) / 128)];
			passEncoder.dispatch(...dispatchGroups);
		}

		{ // SORT TRIANGLES
			let {pipeline} = passes["sort_triangles"];
			passEncoder.setPipeline(pipeline);

			let dispatchGroups = [Math.ceil(numTriangles / 128)];
			passEncoder.dispatch(...dispatchGroups);
		}


		passEncoder.end();
		
		device.queue.submit([commandEncoder.finish()]);

	}

	let pLutGrid = renderer.readBuffer(state.gpu_grids, 4 * lutGridOffset, 4 * lutGridCellCount);
	let pTricountGrid = renderer.readBuffer(state.gpu_grids, 4 * tricountGridOffset, 4 * tricountGridCellCount);
	let pChunks = renderer.readBuffer(state.gpu_chunks, 0, (chunk.index + 1) * 32);

	let [bLutGrid, bTricountGrid, bChunks] 
		= await Promise.all([pLutGrid, pTricountGrid, pChunks]);

	let lutGrid = new Uint32Array(bLutGrid);
	let tricountGrid = new Uint32Array(bTricountGrid);

	state.numChunksProcessed++;
	chunk.voxels.firstVoxel = state.numVoxelsProcessed;
	chunk.voxels.numVoxels = new DataView(bChunks).getUint32(32 * chunk.index + 0, true);
	state.numVoxelsProcessed += chunk.voxels.numVoxels;

	// if(chunk.level <= 2){
	if((chunk.triangles.numIndices / 3) >= maxTrianglesPerNode){
		for(let i = 0; i < 8; i++){
			let numTriangles = tricountGrid[i]
			let triangleOffset = lutGrid[i] - numTriangles;

			if(numTriangles === 0){
				chunk.children[i] = null;
			}else{

				let child = new Chunk();

				child.id = chunk.id + i;
				child.level = chunk.level + 1;
				child.boundingBox = computeChildBox(chunk.boundingBox, i);
				child.parent = chunk;
				child.triangles = {
					gpu_position   : chunk.triangles.gpu_position,
					// gpu_uv         : chunk.triangles.gpu_uv,
					gpu_color      : chunk.triangles.gpu_color,
					gpu_indices    : gpu_sortedIndices,
					firstIndex     : 3 * triangleOffset,
					numIndices     : 3 * numTriangles,
				};
				child.voxels = {
					gpu_voxels     : chunk.voxels.gpu_voxels,
					firstVoxel     : 0,
					numVoxels      : 0,
				};

				chunk.children[i] = child;

				state.chunkList.push(child);

			}
			
		}
	}

	state.currentlyProcessing = null;
	chunk.processed = true;

}

export async function doDownsampling(renderer, node){

	console.log("doDownsampling start", (performance.now() / 1000).toFixed(3));

	let state = getState(renderer, node);

	let root = new Chunk();
	root.id = "r";
	root.boundingBox = node.boundingBox.cube();
	root.triangles = {
		gpu_position   : renderer.getGpuBuffer(node.geometry.findBuffer("position")),
		gpu_color      : renderer.getGpuBuffer(node.geometry.findBuffer("color")),
		gpu_indices    : renderer.getGpuBuffer(node.geometry.indices),
		firstIndex     : 0,
		numIndices     : node.geometry.indices.length,
	};
	root.voxels = {
		gpu_voxels     : state.gpu_voxels,
		firstVoxel     : 0,
		numVoxels      : 0,
	};

	state.chunkList.push(root);

	potree.onUpdate( () => {

		root.traverse((chunk) => {
			chunk.visible = false;
		});

		root.traverse((chunk) => {
			
			let center = chunk.boundingBox.center();
			let size = chunk.boundingBox.size().length();
			let camWorldPos = camera.getWorldPosition();
			let distance = camWorldPos.distanceTo(center);

			let visible = (size / distance) > 0.3 * Potree.settings.debugU;
			if(chunk.id === "r"){
				visible = true;
			} 

			if(visible && chunk.processed === false && state.currentlyProcessing === null){
				processChunk(renderer, node, chunk);

				return false;
			}

			let numTriangles = chunk.triangles.numIndices / 3;
			if(visible && numTriangles < maxTrianglesPerNode){
				let mesh = chunk.triangles;
				potree.renderer.drawMesh(mesh);
			}

			chunk.visible = visible;

		});
	});

	potree.renderer.onDraw( (drawstate) => {
		renderVoxelsLOD(root, drawstate);
	});

	await processChunk(renderer, node, root);

	console.log("doDownsampling end", (performance.now() / 1000).toFixed(3));
	
}