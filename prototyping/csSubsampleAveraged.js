
import {LasLoader, Header} from "LasLoader";
import {Timer} from "potree";

const csSource = `
[[block]] struct F32s {
	values : [[stride(4)]] array<f32>;
};

[[block]] struct U32s {
	values : [[stride(4)]] array<u32>;
};

[[block]] struct U32As {
	values : [[stride(4)]] array<atomic<u32>>;
};

[[block]] struct Result {
	value : atomic<u32>;
};

[[block]] struct SimParams {
	gridSize : f32;
	min_x : f32;
	min_y : f32;
	min_z : f32;
	max_x : f32;
	max_y : f32;
	max_z : f32;
};

[[binding(0), group(0)]] var<uniform> params : SimParams;
[[binding(1), group(0)]] var<storage, read> vertices : F32s;
[[binding(2), group(0)]] var<storage, read_write> result : Result; 

[[binding(5), group(0)]] var<storage, read> in_colors : U32s;

[[binding(10), group(0)]] var<storage, read_write> grid : U32As;
[[binding(11), group(0)]] var<storage, read_write> out_positions : F32s;
[[binding(12), group(0)]] var<storage, read_write> out_colors : U32s;

[[binding(20), group(0)]] var<storage, read_write> grid_sample_positions : F32s;
[[binding(21), group(0)]] var<storage, read_write> grid_sample_colors : U32As;


[[stage(compute), workgroup_size(128)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
	
	var index : u32 = GlobalInvocationID.x;

	var x : f32 = vertices.values[3u * index + 0u];
	var y : f32 = vertices.values[3u * index + 1u];
	var z : f32 = vertices.values[3u * index + 2u];

	var size_x : f32 = params.max_x - params.min_x;
	var size_y : f32 = params.max_y - params.min_y;
	var size_z : f32 = params.max_z - params.min_z;

	var size : f32 = max(max(size_x, size_y), size_z);

	var ix = u32(128.0 * (x - params.min_x) / size);
	var iy = u32(128.0 * (y - params.min_y) / size);
	var iz = u32(128.0 * (z - params.min_z) / size);

	var cellIndex = ix + 128u * iy + 128u * 128u * iz;

	var counter : u32 = atomicAdd(&grid.values[cellIndex], 1u);

	if(counter == 0u){
		var numAccepted : u32 = atomicAdd(&result.value, 1u);

		out_positions.values[3u * numAccepted + 0u] = x;
		out_positions.values[3u * numAccepted + 1u] = y;
		out_positions.values[3u * numAccepted + 2u] = z;

		out_colors.values[numAccepted] = in_colors.values[index];

		grid_sample_positions.values[3u * cellIndex + 0u] = x;
		grid_sample_positions.values[3u * cellIndex + 1u] = y;
		grid_sample_positions.values[3u * cellIndex + 2u] = z;
	}

}
`;

let csSums = `
[[block]] struct F32s {
	values : [[stride(4)]] array<f32>;
};

[[block]] struct U32s {
	values : [[stride(4)]] array<u32>;
};

[[block]] struct U32As {
	values : [[stride(4)]] array<atomic<u32>>;
};

[[block]] struct Result {
	value : atomic<u32>;
};

[[block]] struct SimParams {
	gridSize : f32;
	min_x : f32;
	min_y : f32;
	min_z : f32;
	max_x : f32;
	max_y : f32;
	max_z : f32;
};

[[binding(0), group(0)]] var<uniform> params : SimParams;
[[binding(1), group(0)]] var<storage, read> vertices : F32s;
[[binding(2), group(0)]] var<storage, read_write> result : Result; 

[[binding(5), group(0)]] var<storage, read> in_colors : U32s;

[[binding(10), group(0)]] var<storage, read_write> grid : U32As;
[[binding(11), group(0)]] var<storage, read_write> out_positions : F32s;
[[binding(12), group(0)]] var<storage, read_write> out_colors : U32s;

[[binding(20), group(0)]] var<storage, read_write> grid_sample_positions : F32s;
[[binding(21), group(0)]] var<storage, read_write> grid_sample_colors : U32As;


[[stage(compute), workgroup_size(128)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
	
	var index : u32 = GlobalInvocationID.x;

	var x : f32 = vertices.values[3u * index + 0u];
	var y : f32 = vertices.values[3u * index + 1u];
	var z : f32 = vertices.values[3u * index + 2u];

	var size_x : f32 = params.max_x - params.min_x;
	var size_y : f32 = params.max_y - params.min_y;
	var size_z : f32 = params.max_z - params.min_z;

	var size : f32 = max(max(size_x, size_y), size_z);

	var ix = u32(128.0 * (x - params.min_x) / size);
	var iy = u32(128.0 * (y - params.min_y) / size);
	var iz = u32(128.0 * (z - params.min_z) / size);

	var cellIndex = ix + 128u * iy + 128u * 128u * iz;

	var rgb : u32 = in_colors.values[index];
	var r : u32 = (rgb >>  0u) & 0xFFu;
	var g : u32 = (rgb >>  8u) & 0xFFu;
	var b : u32 = (rgb >> 16u) & 0xFFu;

	var asd = atomicAdd(&grid_sample_colors.values[4u * cellIndex + 0u], r);
	var g3q = atomicAdd(&grid_sample_colors.values[4u * cellIndex + 1u], g);
	var dh5 = atomicAdd(&grid_sample_colors.values[4u * cellIndex + 2u], b);
	var n5g = atomicAdd(&grid_sample_colors.values[4u * cellIndex + 3u], 1u);

}
`;

let csGridToVbo = `
[[block]] struct F32s {
	values : [[stride(4)]] array<f32>;
};

[[block]] struct U32s {
	values : [[stride(4)]] array<u32>;
};

[[block]] struct U32As {
	values : [[stride(4)]] array<atomic<u32>>;
};

[[block]] struct Result {
	value : atomic<u32>;
};

[[block]] struct SimParams {
	gridSize : f32;
	min_x : f32;
	min_y : f32;
	min_z : f32;
	max_x : f32;
	max_y : f32;
	max_z : f32;
};

[[binding(0), group(0)]] var<uniform> params : SimParams;
[[binding(1), group(0)]] var<storage, read> vertices : F32s;
[[binding(2), group(0)]] var<storage, read_write> result : Result; 

[[binding(5), group(0)]] var<storage, read> in_colors : U32s;

[[binding(10), group(0)]] var<storage, read_write> grid : U32As;
[[binding(11), group(0)]] var<storage, read_write> out_positions : F32s;
[[binding(12), group(0)]] var<storage, read_write> out_colors : U32s;

[[binding(20), group(0)]] var<storage, read_write> grid_sample_positions : F32s;
[[binding(21), group(0)]] var<storage, read_write> grid_sample_colors : U32s;
// [[binding(22), group(0)]] var<storage, read_write> vbo_counter : U32As;


[[stage(compute), workgroup_size(128)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
	
	var index : u32 = GlobalInvocationID.x;

	var R     = grid_sample_colors.values[4u * index + 0u];
	var G     = grid_sample_colors.values[4u * index + 1u];
	var B     = grid_sample_colors.values[4u * index + 2u];
	var count = grid_sample_colors.values[4u * index + 3u];

	// if(count > 0u){

	// 	var x = grid_sample_positions[3u * index + 0u];
	// 	var y = grid_sample_positions[3u * index + 1u];
	// 	var z = grid_sample_positions[3u * index + 2u];

	// 	var r = R / count;
	// 	var g = G / count;
	// 	var b = B / count;
	// 	var rgb = r | (g << 8) | (b << 16);



	// }


}
`;

function createPipeline(device, source){
	let pipeline = device.createComputePipeline({
		compute: {
			module: device.createShaderModule({code: source}),
			entryPoint: 'main',
		},
	});

	return pipeline;
}

export async function csSubsample(las, renderer){

	let {device} = renderer;

	let storage_flags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
	let uniform_flags = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
	const usage = storage_flags;

	const uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});

	const resultBuffer        = device.createBuffer({usage, size: 256});
	const gridBuffer          = device.createBuffer({usage: GPUBufferUsage.STORAGE, size: 200_000_000});
	// const gridBuffer          = device.createBuffer({usage, size: 4 * 128 ** 3});
	const outPositionsBuffer  = device.createBuffer({usage, size: 100_000_000});
	const outColorsBuffer     = device.createBuffer({usage, size: 10_000_000 * 4});
	const gridSamplePositions = device.createBuffer({usage, size: 12 * 128 ** 3});
	const gridSampleColors    = device.createBuffer({usage, size: 16 * 128 ** 3});
	// const atomicCounters      = device.createBuffer({usage, size: 256});

	// const wrgt  = device.createBuffer({usage, size: 140_000_000});

	let pipeline          = createPipeline(device, csSource);
	let pipelineSums      = createPipeline(device, csSums);
	// let pipelineGridToVbo = createPipeline(device, csGridToVbo);

	let vboPosition = renderer.getGpuBuffer(las.buffers.positionf32);
	let vboColor = renderer.getGpuBuffer(las.buffers.color);

	let bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: {buffer: vboPosition}},
			{binding: 2, resource: {buffer: resultBuffer}},
			{binding: 5, resource: {buffer: vboColor}},
			{binding: 10, resource: {buffer: gridBuffer}},
			{binding: 11, resource: {buffer: outPositionsBuffer}},
			{binding: 12, resource: {buffer: outColorsBuffer}},
			{binding: 20, resource: {buffer: gridSamplePositions}},
			{binding: 21, resource: {buffer: gridSampleColors}},
			// {binding: 22, resource: {buffer: atomicCounters}},
		],
	});

	let min = las.header.min;
	let max = las.header.max;
	device.queue.writeBuffer(
		uniformBuffer,
		0,
		new Float32Array([
			1.0,
			min.x, min.y, min.z, 
			max.x, max.y, max.z, 
		])
	);

	{ // SAMPLE
		let commandEncoder = device.createCommandEncoder();
		let passEncoder = commandEncoder.beginComputePass();

		Timer.timestamp(passEncoder,"sample-start");
		
		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, bindGroup);
		let numPoints = las.header.numPoints;
		let groups = Math.ceil(numPoints / 128);
		passEncoder.dispatch(groups);

		Timer.timestamp(passEncoder,"sample-end", {print: true});
		
		passEncoder.endPass();
		device.queue.submit([commandEncoder.finish()]);
	}

	{ // SUMS
		let commandEncoder = device.createCommandEncoder();
		let passEncoder = commandEncoder.beginComputePass();

		Timer.timestamp(passEncoder,"sums-start");
		
		passEncoder.setPipeline(pipelineSums);
		passEncoder.setBindGroup(0, bindGroup);
		let numPoints = las.header.numPoints;
		let groups = Math.ceil(numPoints / 128);
		passEncoder.dispatch(groups);

		Timer.timestamp(passEncoder,"sums-end", {print: true});
		
		passEncoder.endPass();
		device.queue.submit([commandEncoder.finish()]);
	}

	// { // GRID TO VBO
	// 	let commandEncoder = device.createCommandEncoder();
	// 	let passEncoder = commandEncoder.beginComputePass();

	// 	Timer.timestamp(passEncoder,"sums-start");
		
	// 	passEncoder.setPipeline(pipelineGridToVbo);
	// 	passEncoder.setBindGroup(0, bindGroup);
	// 	let numPoints = las.header.numPoints;
	// 	let groups = Math.ceil(numPoints / 128);
	// 	passEncoder.dispatch(groups);

	// 	Timer.timestamp(passEncoder,"sums-end", {print: true});
		
	// 	passEncoder.endPass();
	// 	device.queue.submit([commandEncoder.finish()]);
	// }

	let result = await renderer.readBuffer(resultBuffer, 0, 4);
	let numAccepted = new Uint32Array(result)[0];
	
	console.log(numAccepted);

	let result1 = await renderer.readBuffer(outPositionsBuffer, 0, 12 * numAccepted);
	let resultColor = await renderer.readBuffer(outColorsBuffer, 0, 4 * numAccepted);

	{
		let header = new Header();
		header.numPoints = numAccepted;
		header.scale = las.header.scale.clone();
		header.offset = las.header.scale.clone();
		header.min = las.header.min;
		header.max = las.header.max;

		let positionf32 = new Float32Array(result1);
		let color = new Uint8Array(resultColor);

		let buffers = {positionf32, color};

		return {header, buffers};

	}
}

