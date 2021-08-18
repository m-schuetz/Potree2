
import { Vector3 } from "../../math/Vector3.js";
import {voxelGridSize, chunkGridSize, toIndex1D, toIndex3D} from "./common.js";
import {storage_flags, uniform_flags} from "./common.js";
import { generateVoxelsCompute } from "./generate_voxels_compute.js";
import {renderVoxels} from "./renderVoxels.js";
import {SPECTRAL} from "potree";

export let csDownsampling = `

[[block]] struct Uniforms {
	numTriangles     : u32;
	gridSize         : u32;
	firstTriangle    : u32;
	pad2             : u32;
	bbMin            : vec4<f32>;      // offset(16)
	bbMax            : vec4<f32>;      // offset(32)
	chunkIndex       : u32;            // offset(48)
	level            : u32;            // offset(52)
};

[[block]] struct Metadata {
	offsetCounter   : atomic<u32>;
	numVoxelsAdded  : atomic<u32>;
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

struct Chunk{
	level        : u32;
	min_x        : f32;
	min_y        : f32;
	min_z        : f32;
	firstVoxel   : atomic<i32>;
	numVoxels    : atomic<u32>;
	pad1         : u32;
	pad2         : u32;
};

[[block]] struct F32s { values : [[stride(4)]] array<f32>; };
[[block]] struct U32s { values : [[stride(4)]] array<u32>; };
[[block]] struct I32s { values : [[stride(4)]] array<i32>; };
[[block]] struct AU32s { values : [[stride(4)]] array<atomic<u32>>; };
[[block]] struct AI32s { values : [[stride(4)]] array<atomic<i32>>; };

[[block]] struct Chunks { values : [[stride(32)]] array<Chunk>; };
[[block]] struct Voxels { values : [[stride(32)]] array<Voxel>; };

// IN
[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(10), group(0)]] var<storage, read_write> indices    : U32s;
[[binding(11), group(0)]] var<storage, read_write> positions  : F32s;
[[binding(12), group(0)]] var<storage, read_write> colors     : U32s;

// OUT
[[binding(20), group(0)]] var<storage, read_write> voxelGrid  : AU32s;
[[binding(21), group(0)]] var<storage, read_write> chunks     : Chunks;
[[binding(22), group(0)]] var<storage, read_write> voxels     : Voxels;

[[binding(50), group(0)]] var<storage, read_write> metadata : Metadata;

fn toVoxelPos(position : vec3<f32>) -> vec3<f32>{

	var bbMin = vec3<f32>(uniforms.bbMin.x, uniforms.bbMin.y, uniforms.bbMin.z);
	var bbMax = vec3<f32>(uniforms.bbMax.x, uniforms.bbMax.y, uniforms.bbMax.z);
	var bbSize = bbMax - bbMin;
	var cubeSize = max(max(bbSize.x, bbSize.y), bbSize.z);
	var gridSize = f32(uniforms.gridSize);

	var gx = gridSize * (position.x - uniforms.bbMin.x) / cubeSize;
	var gy = gridSize * (position.y - uniforms.bbMin.y) / cubeSize;
	var gz = gridSize * (position.z - uniforms.bbMin.z) / cubeSize;

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
	
	var g42 = uniforms.numTriangles;
	var kj6 = metadata.value1;
	var rwg = indices.values[0];
	var rb5 = positions.values[0];
	var lw5 = colors.values[0];
	var g55 = atomicLoad(&voxelGrid.values[0]);
	var a10 = chunks.values[0].level;
	var a20 = voxels.values[0];
	
}

[[stage(compute), workgroup_size(128)]]
fn main_accumulate([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	if(GlobalInvocationID.x >= uniforms.numTriangles){
		return;
	}

	var triangleIndex = GlobalInvocationID.x + uniforms.firstTriangle;

	doIgnore();

	var i0 = indices.values[3u * triangleIndex + 0u];
	var i1 = indices.values[3u * triangleIndex + 1u];
	var i2 = indices.values[3u * triangleIndex + 2u];

	var p0 = loadPosition(i0);
	var p1 = loadPosition(i1);
	var p2 = loadPosition(i2);
	var center = (p0 + p1 + p2) / 3.0;

	var voxelPos = toVoxelPos(center);
	var voxelIndex = toIndex1D(uniforms.gridSize, voxelPos);

	var color = (loadColor(i0) + loadColor(i1) + loadColor(i2)) / 3u;

	var acefg1 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 0u], color.x);
	var acefg2 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 1u], color.y);
	var acefg3 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 2u], color.z);
	var acefg4 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 3u], 1u);

}


[[stage(compute), workgroup_size(128)]]
fn main_gather([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	var voxelIndex = GlobalInvocationID.x;

	doIgnore();

	var maxVoxels = uniforms.gridSize * uniforms.gridSize * uniforms.gridSize;
	if(voxelIndex >= maxVoxels){
		return;
	}
	
	var count = atomicLoad(&voxelGrid.values[4u * voxelIndex + 3u]);

	if(voxelIndex == 0u){

		chunks.values[uniforms.chunkIndex].level = uniforms.level;
		chunks.values[uniforms.chunkIndex].min_x = uniforms.bbMin.x;
		chunks.values[uniforms.chunkIndex].min_z = uniforms.bbMin.z;
		chunks.values[uniforms.chunkIndex].min_y = uniforms.bbMin.y;
	}


	if(count == 0u){
		return;
	}

	var r = atomicLoad(&voxelGrid.values[4u * voxelIndex + 0u]) / count;
	var g = atomicLoad(&voxelGrid.values[4u * voxelIndex + 1u]) / count;
	var b = atomicLoad(&voxelGrid.values[4u * voxelIndex + 2u]) / count;

	var voxelCoord = toIndex3D(uniforms.gridSize, voxelIndex);

	var voxel = Voxel();

	var bbMin = vec3<f32>(uniforms.bbMin.x, uniforms.bbMin.y, uniforms.bbMin.z);
	var bbMax = vec3<f32>(uniforms.bbMax.x, uniforms.bbMax.y, uniforms.bbMax.z);
	var bbSize = bbMax - bbMin;
	var cubeSize = max(max(bbSize.x, bbSize.y), bbSize.z);
	voxel.x = cubeSize * (f32(voxelCoord.x) / f32(uniforms.gridSize)) + uniforms.bbMin.x;
	voxel.y = cubeSize * (f32(voxelCoord.y) / f32(uniforms.gridSize)) + uniforms.bbMin.y;
	voxel.z = cubeSize * (f32(voxelCoord.z) / f32(uniforms.gridSize)) + uniforms.bbMin.z;
	voxel.size = cubeSize / f32(uniforms.gridSize);
	voxel.r = r;
	voxel.g = g;
	voxel.b = b;
	voxel.count = count;

	var voxelOffset = atomicAdd(&metadata.numVoxelsAdded, 1u);
	voxels.values[voxelOffset] = voxel;

	var fgh24 = atomicMin(&chunks.values[uniforms.chunkIndex].firstVoxel, i32(voxelOffset) - 123456789);
	var vrw = atomicAdd(&chunks.values[uniforms.chunkIndex].numVoxels, 1u);

}


`;


export async function doDownsampling(renderer, node, result_chunking){

	let {device} = renderer;

	let accumulateGridSize = 16 * voxelGridSize ** 3;
	let chunksSize = 32 * 100_000;
	let voxelsSize = 32 * 4_000_000;
	let gpu_accumulate = device.createBuffer({size: accumulateGridSize, usage: storage_flags});
	let gpu_chunks = device.createBuffer({size: chunksSize, usage: storage_flags});
	let gpu_voxels = device.createBuffer({size: voxelsSize, usage: storage_flags});
	let gpu_meta = device.createBuffer({size: 256, usage: storage_flags});
	let numVoxelsAdded = 0;

	console.time("downsampling");

	potree.renderer.onDraw( (drawstate) => {
		let voxels = {gpu_meta, gpu_voxels, numVoxels: numVoxelsAdded};

		renderVoxels(drawstate, voxels);
	});

	let numChunksProcessed = 0;
	for(let chunkIndex = 0; chunkIndex < result_chunking.chunks.length; chunkIndex++){
		let chunk = result_chunking.chunks[chunkIndex];

		let numTriangles = chunk.numTriangles;
		let triangleOffset = chunk.triangleOffset;

		let cube = chunk.boundingBox;
		let cubeSize = cube.max.x - cube.min.x;

		renderer.fillBuffer(gpu_accumulate, 0, 4 * voxelGridSize ** 3);


		let uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});
		{
			let buffer = new ArrayBuffer(256);
			let view = new DataView(buffer);

			view.setUint32(0, numTriangles, true);
			view.setUint32(4, voxelGridSize, true);
			view.setUint32(8, triangleOffset, true);

			view.setFloat32(16, cube.min.x, true);
			view.setFloat32(20, cube.min.y, true);
			view.setFloat32(24, cube.min.z, true);

			view.setFloat32(32, cube.max.x, true);
			view.setFloat32(36, cube.max.y, true);
			view.setFloat32(40, cube.max.z, true);

			view.setUint32(48, chunkIndex, true);
			view.setUint32(52, chunk.level, true);

			device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
		}

		let gpu_indices = result_chunking.gpu_sortedIndices;
		let host_positions = node.geometry.findBuffer("position");
		let host_colors = node.geometry.findBuffer("color");
		let gpu_positions = renderer.getGpuBuffer(host_positions);
		let gpu_colors = renderer.getGpuBuffer(host_colors);
		

		let bindGroups = [
			{
				location: 0,
				entries: [
					{binding:  0, resource: {buffer: uniformBuffer}},
					{binding: 10, resource: {buffer: gpu_indices}},
					{binding: 11, resource: {buffer: gpu_positions}},
					{binding: 12, resource: {buffer: gpu_colors}},

					{binding: 20, resource: {buffer: gpu_accumulate}},
					{binding: 21, resource: {buffer: gpu_chunks}},
					{binding: 22, resource: {buffer: gpu_voxels}},

					{binding: 50, resource: {buffer: gpu_meta}},
				],
			}
		];

		renderer.runCompute({
			code: csDownsampling,
			entryPoint: "main_accumulate",
			bindGroups: bindGroups,
			dispatchGroups: [Math.ceil(numTriangles / 128)],
		});

		renderer.runCompute({
			code: csDownsampling,
			entryPoint: "main_gather",
			bindGroups: bindGroups,
			dispatchGroups: [Math.ceil((voxelGridSize ** 3) / 128)],
		});

		if((chunkIndex % 10) === 0){
			let pMetadata = renderer.readBuffer(gpu_meta, 0, 32);

			let [rMetadata] = await Promise.all([pMetadata]);

			numVoxelsAdded = new Uint32Array(rMetadata)[1];
			console.log("numVoxelsAdded: ", numVoxelsAdded);
		}

		numChunksProcessed++;
	}

	{
		let rChunks = await renderer.readBuffer(gpu_chunks, 0, numChunksProcessed * 32);
		let view = new DataView(rChunks);

		let chunks = [];
		for(let i = 0; i < numChunksProcessed; i++){
			let level = view.getUint32(32 * i + 0, true);
			let minX = view.getFloat32(32 * i + 4, true);
			let minY = view.getFloat32(32 * i + 8, true);
			let minZ = view.getFloat32(32 * i + 12, true);
			let firstVoxel = view.getInt32(32 * i + 16, true);
			let numVoxels = view.getUint32(32 * i + 20, true);

			let chunk = {
				position: new Vector3(minX, minY, minZ),
				firstVoxel, numVoxels
			};

			chunks.push(chunk);
		}

		potree.onUpdate( () => {

			let cube = node.boundingBox.cube();
			let cubeSize = cube.max.x - cube.min.x;
			let chunkSize = cubeSize / chunkGridSize;

			for(let chunk of chunks){
				let position = chunk.position.clone().addScalar(chunkSize / 2);
				let scale = new Vector3(1, 1, 1).multiplyScalar(chunkSize);
				let color = new Vector3(Math.min(chunk.numVoxels / 2, 255), 0, 0);
				// let color = new Vector3(...SPECTRAL.get(chunk.numVoxels / 255)).multiplyScalar(255);
				potree.renderer.drawBoundingBox(position, scale, color);
			}
		});
	}

	console.timeEnd("downsampling");

	

	
}