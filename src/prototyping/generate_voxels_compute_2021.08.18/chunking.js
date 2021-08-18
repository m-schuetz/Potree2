
import {Box3, Vector3} from "potree";
import {chunkGridSize, toIndex1D, toIndex3D} from "./common.js";
import {storage_flags, uniform_flags} from "./common.js";

let csChunking = `

[[block]] struct Uniforms {
	numTriangles     : u32;
	gridSize         : u32;
	pad1             : u32;
	pad2             : u32;
	bbMin            : vec3<f32>;      // offset(16)
	bbMax            : vec3<f32>;      // offset(32)
};

[[block]] struct Dbg {
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
[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(10), group(0)]] var<storage, read_write> indices : U32s;
[[binding(11), group(0)]] var<storage, read_write> positions : F32s;
[[binding(12), group(0)]] var<storage, read_write> colors : U32s;


// OUT
[[binding(20), group(0)]] var<storage, read_write> tricountGrid : AU32s;
[[binding(21), group(0)]] var<storage, read_write> LutGrid : AU32s;
[[binding(22), group(0)]] var<storage, read_write> voxelGrid : AU32s;
[[binding(23), group(0)]] var<storage, read_write> sortedIndices : U32s;

// DEBUG
[[binding(50), group(0)]] var<storage, read_write> dbg : Dbg;

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

fn doIgnore(){
	
	var g42 = uniforms.numTriangles;
	var kj6 = dbg.value1;
	var b53 = atomicLoad(&counters.values[0]);
	var rwg = indices.values[0];
	var rb5 = positions.values[0];
	var lw5 = colors.values[0];
	// var n52 = uvs.values[0];
	var g55 = atomicLoad(&LUT.values[0]);
	var a35 = sortedBuffer.values[0];
	
}

[[stage(compute), workgroup_size(128)]]
fn main_count([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	var triangleIndex = GlobalInvocationID.x;

	if(triangleIndex >= uniforms.numTriangles){
		return;
	}

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

	var acefg = atomicAdd(&counters.values[voxelIndex], 1u);

	if(triangleIndex == 0u){
		dbg.value0 = uniforms.gridSize;
		// dbg.value1 = voxelPos.y;
		// dbg.value2 = voxelPos.z;

		dbg.value_f32_0 = center.x;
		dbg.value_f32_1 = center.y;
		dbg.value_f32_2 = center.z;
	}

}

[[stage(compute), workgroup_size(128)]]
fn main_create_lut([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	var voxelIndex = GlobalInvocationID.x;

	doIgnore();

	var maxVoxels = uniforms.gridSize * uniforms.gridSize * uniforms.gridSize;
	if(voxelIndex >= maxVoxels){
		return;
	}

	var numTriangles = atomicLoad(&counters.values[voxelIndex]);

	var offset = 0u;
	if(numTriangles > 0u){
		offset = atomicAdd(&dbg.offsetCounter, numTriangles);
	}

	// atomicStore(&LUT.values[voxelIndex], offset);
	var be5t = atomicExchange(&LUT.values[voxelIndex], offset);

}

[[stage(compute), workgroup_size(128)]]
fn main_sort_triangles([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	var triangleIndex = GlobalInvocationID.x;

	doIgnore();

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

	var voxelPos = toVoxelPos(center);
	var voxelIndex = toIndex1D(uniforms.gridSize, voxelPos);

	var triangleOffset = atomicAdd(&LUT.values[voxelIndex], 1u);

	sortedBuffer.values[3u * triangleOffset + 0u] = i0;
	sortedBuffer.values[3u * triangleOffset + 1u] = i1;
	sortedBuffer.values[3u * triangleOffset + 2u] = i2;

	// colors.values[i0] = 123u * voxelIndex;
	// colors.values[i1] = 123u * voxelIndex;
	// colors.values[i2] = 123u * voxelIndex;



}


`;

export async function doChunking(renderer, node){

	let {device} = renderer;

	let numTriangles = node.geometry.indices.length / 3;
	let box = node.boundingBox.clone();
	let cube = box.cube();

	let uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});
	{
		let buffer = new ArrayBuffer(256);
		let view = new DataView(buffer);

		view.setUint32(0, numTriangles, true);
		view.setUint32(4, chunkGridSize, true);

		view.setFloat32(16, cube.min.x, true);
		view.setFloat32(20, cube.min.y, true);
		view.setFloat32(24, cube.min.z, true);

		view.setFloat32(32, cube.max.x, true);
		view.setFloat32(36, cube.max.y, true);
		view.setFloat32(40, cube.max.z, true);

		device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
	}

	let counterGridSize = 4 * chunkGridSize ** 3;
	let gpu_counters = device.createBuffer({size: counterGridSize, usage: storage_flags});
	let gpu_LUT = device.createBuffer({size: counterGridSize, usage: storage_flags});
	let gpu_indices = renderer.getGpuBuffer(node.geometry.indices);
	let host_positions = node.geometry.findBuffer("position");
	let host_colors = node.geometry.findBuffer("color");
	let gpu_positions = renderer.getGpuBuffer(host_positions);
	let gpu_colors = renderer.getGpuBuffer(host_colors);
	let gpu_dbg = device.createBuffer({size: 256, usage: storage_flags});
	let gpu_sortedIndices = device.createBuffer({size: 4 * 3 * numTriangles, usage: storage_flags});

	let bindGroups = [
		{
			location: 0,
			entries: [
				{binding:  0, resource: {buffer: uniformBuffer}},
				{binding: 10, resource: {buffer: gpu_indices}},
				{binding: 11, resource: {buffer: gpu_positions}},
				{binding: 12, resource: {buffer: gpu_colors}},
				// {binding: 13, resource: {buffer: gpu_uvs}},
				{binding: 20, resource: {buffer: gpu_counters}},
				{binding: 21, resource: {buffer: gpu_LUT}},
				{binding: 22, resource: {buffer: gpu_sortedIndices}},
				{binding: 50, resource: {buffer: gpu_dbg}},
			],
		}
	];


	renderer.runCompute({
		code: csChunking,
		entryPoint: "main_count",
		bindGroups: bindGroups,
		dispatchGroups: [Math.ceil(numTriangles / 128)],
	});

	renderer.runCompute({
		code: csChunking,
		entryPoint: "main_create_lut",
		bindGroups: bindGroups,
		dispatchGroups: [Math.ceil((chunkGridSize ** 3) / 128)],
	});

	renderer.runCompute({
		code: csChunking,
		entryPoint: "main_sort_triangles",
		bindGroups: bindGroups,
		dispatchGroups: [Math.ceil(numTriangles / 128)],
	});

	let pDebug = renderer.readBuffer(gpu_dbg, 0, 32);
	let pCounters = renderer.readBuffer(gpu_counters, 0, counterGridSize);
	let pLUT = renderer.readBuffer(gpu_LUT, 0, counterGridSize);
	let pSortedIndices = renderer.readBuffer(gpu_sortedIndices, 0, 4 * 3 * numTriangles);

	let [
		rDebug, rCounters, rLUT, rSortedIndices
	] = await Promise.all([pDebug, pCounters, pLUT, pSortedIndices]);

	let chunks = [];
	{
		let u32Counters = new Uint32Array(rCounters);
		let u32LUT = new Uint32Array(rLUT);

		for(let voxelIndex = 0; voxelIndex < chunkGridSize ** 3; voxelIndex++){
			let coord = toIndex3D(chunkGridSize, voxelIndex);
			let numTriangles = u32Counters[voxelIndex];
			let triangleOffset = u32LUT[voxelIndex] - numTriangles;

			if(numTriangles === 0){
				continue;
			}

			let level = Math.log2(chunkGridSize);
			let cubeSize = cube.max.x - cube.min.x;
			let chunkSize = cubeSize / (2 ** level);
			let min = new Vector3(
				cubeSize * (coord.x / chunkGridSize) + cube.min.x,
				cubeSize * (coord.y / chunkGridSize) + cube.min.y,
				cubeSize * (coord.z / chunkGridSize) + cube.min.z,
			);
			let max = min.clone().addScalar(chunkSize);
			let boundingBox = new Box3(min, max);

			let chunk = {level, boundingBox, coord, numTriangles, triangleOffset};

			chunks.push(chunk);
		}

		console.log("chunks: ");
		console.log(chunks);
	}

	let result = {
		chunks,

		rDebug, 
		rCounters, 
		rLUT, 
		rSortedIndices,

		gpu_counters,
		gpu_LUT,
		gpu_indices,
		gpu_positions,
		gpu_colors,
		gpu_dbg,
		gpu_sortedIndices,
	};

	return result;
}


