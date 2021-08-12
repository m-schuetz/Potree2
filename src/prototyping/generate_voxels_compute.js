
let gridSize = 64;

let cs = `

[[block]] struct Uniforms {
	numTriangles     : u32;
	gridSize         : u32;
	pad1             : u32;
	pad2             : u32;
	bbMin            : vec3<f32>;      // offset(16)
	bbMax            : vec3<f32>;      // offset(32)
};

[[block]] struct Dbg {
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
[[block]] struct AU32s { values : [[stride(4)]] array<atomic<u32>>; };

// IN
[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(10), group(0)]] var<storage, read_write> indices : U32s;
[[binding(11), group(0)]] var<storage, read_write> positions : F32s;

// OUT
[[binding(20), group(0)]] var<storage, read_write> counters : AU32s;

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

fn loadPosition(vertexIndex : u32) -> vec3<f32> {
	
	var position = vec3<f32>(
		positions.values[3u * vertexIndex + 0u],
		positions.values[3u * vertexIndex + 1u],
		positions.values[3u * vertexIndex + 2u],
	);

	return position;
};

[[stage(compute), workgroup_size(128)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

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
`;

let storage_flags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
let uniform_flags = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

export async function generateVoxelsCompute(renderer, node){

	let {device} = renderer;

	let numTriangles = node.geometry.indices.length / 3;
	let box = node.boundingBox.clone();
	let cube = box.cube();

	let uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});
	{
		let buffer = new ArrayBuffer(256);
		let view = new DataView(buffer);

		view.setUint32(0, numTriangles, true);
		view.setUint32(4, gridSize, true);

		view.setFloat32(16, cube.min.x, true);
		view.setFloat32(20, cube.min.y, true);
		view.setFloat32(24, cube.min.z, true);

		view.setFloat32(32, cube.max.x, true);
		view.setFloat32(36, cube.max.y, true);
		view.setFloat32(40, cube.max.z, true);

		device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
	}

	let counterGridSize = 4 * gridSize ** 3;
	let gpu_counters = device.createBuffer({size: counterGridSize, usage: storage_flags});
	let gpu_indices = renderer.getGpuBuffer(node.geometry.indices);
	let host_positions = node.geometry.findBuffer("position");
	let gpu_positions = renderer.getGpuBuffer(host_positions);
	let gpu_dbg = device.createBuffer({size: 256, usage: storage_flags});

	renderer.runCompute({
		code: cs,
		bindGroups: [
			{
				location: 0,
				entries: [
					{binding:  0, resource: {buffer: uniformBuffer}},
					{binding: 10, resource: {buffer: gpu_indices}},
					{binding: 11, resource: {buffer: gpu_positions}},
					{binding: 20, resource: {buffer: gpu_counters}},
					{binding: 50, resource: {buffer: gpu_dbg}},
				],
			}
		],
		dispatchGroups: [Math.ceil(numTriangles / 128)],
	});

	renderer.readBuffer(gpu_dbg, 0, 32).then(result => {
		console.log(new Uint32Array(result, 0, 4));
		console.log(new Float32Array(result, 16, 4));
	});

	renderer.readBuffer(gpu_counters, 0, counterGridSize).then(result => {
		
		let sum = 0;
		let numVoxels = 0;
		let u32 = new Uint32Array(result);

		for(let i = 0; i < u32.length; i++){
			sum += u32[i];
			numVoxels += u32[i] > 0 ? 1 : 0;
		}

		console.log("sum: ", sum);
		console.log("numVoxels: ", numVoxels);

	});

}