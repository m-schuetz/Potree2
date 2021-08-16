
import {voxelGridSize, toIndex1D, toIndex3D} from "./common.js";
import {storage_flags, uniform_flags} from "./common.js";
import { generateVoxelsCompute } from "./generate_voxels_compute.js";

export let csDownsampling = `

[[block]] struct Uniforms {
	numTriangles     : u32;
	gridSize         : u32;
	firstTriangle    : u32;
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
[[binding(20), group(0)]] var<storage, read_write> voxelGrid : AU32s;

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
	var kj6 = dbg.value1;
	var rwg = indices.values[0];
	var rb5 = positions.values[0];
	var lw5 = colors.values[0];
	var g55 = atomicLoad(&voxelGrid.values[0]);

	// var b53 = atomicLoad(&counters.values[0]);
	// var n52 = uvs.values[0];
	// var g55 = atomicLoad(&LUT.values[0]);
	// var a35 = sortedBuffer.values[0];
	
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

	// var color = loadColor(0u);
	var color = (loadColor(i0) + loadColor(i1) + loadColor(i2)) / 3u;

	// var uColor = colors.values[0u];
	// var uColor = colors.values[i0];
	// var R = (uColor >>  0u) & 0xFFu;
	// var G = (uColor >>  8u) & 0xFFu;
	// var B = (uColor >> 16u) & 0xFFu;
	// var acefg1 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 0u], R);
	// var acefg2 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 1u], G);
	// var acefg3 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 2u], B);
	// var acefg4 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 3u], 1u);

	var acefg1 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 0u], color.x);
	var acefg2 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 1u], color.y);
	var acefg3 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 2u], color.z);
	var acefg4 = atomicAdd(&voxelGrid.values[4u * voxelIndex + 3u], 1u);

	

	if(GlobalInvocationID.x == 0u){
		dbg.value0 = triangleIndex;
		dbg.value1 = i0;
		dbg.value2 = i1;
		dbg.value3 = i2;
	}

}



`;


export async function doDownsampling(renderer, node, result_chunking){

	let {device} = renderer;

	let chunk = result_chunking.chunks[322];
	let numTriangles = chunk.numTriangles;
	let triangleOffset = chunk.triangleOffset;

	let cube = chunk.boundingBox;


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

		device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
	}

	let accumulateGridSize = 16 * voxelGridSize ** 3;
	let gpu_accumulate = device.createBuffer({size: accumulateGridSize, usage: storage_flags});
	// let gpu_indices = renderer.getGpuBuffer(node.geometry.indices);
	let gpu_indices = result_chunking.gpu_sortedIndices;
	let host_positions = node.geometry.findBuffer("position");
	let host_colors = node.geometry.findBuffer("color");
	let gpu_positions = renderer.getGpuBuffer(host_positions);
	let gpu_colors = renderer.getGpuBuffer(host_colors);
	let gpu_dbg = device.createBuffer({size: 256, usage: storage_flags});

	let bindGroups = [
		{
			location: 0,
			entries: [
				{binding:  0, resource: {buffer: uniformBuffer}},
				{binding: 10, resource: {buffer: gpu_indices}},
				{binding: 11, resource: {buffer: gpu_positions}},
				{binding: 12, resource: {buffer: gpu_colors}},
				{binding: 20, resource: {buffer: gpu_accumulate}},
				{binding: 50, resource: {buffer: gpu_dbg}},
			],
		}
	];

	renderer.runCompute({
		code: csDownsampling,
		entryPoint: "main_accumulate",
		bindGroups: bindGroups,
		dispatchGroups: [Math.ceil(numTriangles / 128)],
	});



	let pDebug = renderer.readBuffer(gpu_dbg, 0, 32);
	let pAccumulate = renderer.readBuffer(gpu_accumulate, 0, accumulateGridSize);

	let [rDebug, rAccumulate] = await Promise.all([pDebug, pAccumulate]);
	
	let u32Accumulate = new Uint32Array(rAccumulate);
	console.log(u32Accumulate);

	let voxels = [];
	for(let voxelIndex = 0; voxelIndex < voxelGridSize ** 3; voxelIndex++){
		let coord = toIndex3D(voxelGridSize, voxelIndex);
		let cubeSize = cube.max.x - cube.min.x;
		let voxelSize = cubeSize / voxelGridSize;

		let count = u32Accumulate[4 * voxelIndex + 3];

		if(count === 0){
			continue;
		}

		let position = new Vector3(
			cubeSize * (coord.x / voxelGridSize) + cube.min.x + voxelSize / 2,
			cubeSize * (coord.y / voxelGridSize) + cube.min.y + voxelSize / 2,
			cubeSize * (coord.z / voxelGridSize) + cube.min.z + voxelSize / 2,
		);
		let scale = new Vector3(voxelSize, voxelSize, voxelSize);
		let color = new Vector3(
			Math.floor(u32Accumulate[4 * voxelIndex + 0] / count), 
			Math.floor(u32Accumulate[4 * voxelIndex + 1] / count), 
			Math.floor(u32Accumulate[4 * voxelIndex + 2] / count), 
		);

		let voxel = {position, scale, color};
		voxels.push(voxel);
	}

	potree.onUpdate( () => {
		for(let voxel of voxels){
			potree.renderer.drawBox(voxel.position, voxel.scale, voxel.color);
		}
	});

	let triangleIndex = new Uint32Array(rDebug, 16, 4)[0];
	let i0 = new Uint32Array(rDebug, 16, 4)[1];
	let i1 = new Uint32Array(rDebug, 16, 4)[2];
	let i2 = new Uint32Array(rDebug, 16, 4)[3];

	// console.log({triangleIndex, i0, i1, i2});



	
}