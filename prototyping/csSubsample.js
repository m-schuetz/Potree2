
import {LasLoader, Header} from "LasLoader";
import {Timer} from "potree";

const csSource = `
struct F32s {
	values : array<f32>;
};

struct U32s {
	values : array<atomic<u32>>;
};

struct Result {
	value : atomic<u32>;
};

struct SimParams {
	gridSize : f32;
	min_x : f32;
	min_y : f32;
	min_z : f32;
	max_x : f32;
	max_y : f32;
	max_z : f32;
};

@binding(0) @group(0) var<uniform> params : SimParams;
@binding(1) @group(0) var<storage, read> vertices : F32s;
@binding(2) @group(0) var<storage, read_write> result : Result; 

[[binding(5), group(0)]] var<storage, read> in_colors : U32s;

[[binding(10), group(0)]] var<storage, read_write> grid : U32s;
[[binding(11), group(0)]] var<storage, read_write> out_positions : F32s;
[[binding(12), group(0)]] var<storage, read_write> out_colors : U32s;


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
	}

}
`;


export async function csSubsample(las, renderer){

	let {device} = renderer;

	let storage_flags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
	let uniform_flags = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
	const usage = storage_flags;

	const uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});

	const resultBuffer        = device.createBuffer({usage, size: 256});
	const gridBuffer          = device.createBuffer({usage, size: 4 * 128 ** 3});
	const outPositionsBuffer  = device.createBuffer({usage, size: 10_000_000 * 12});
	const outColorsBuffer     = device.createBuffer({usage, size: 10_000_000 * 4});

	let pipeline = device.createComputePipeline({
		compute: {
			module: device.createShaderModule({
				code: csSource,
			}),
			entryPoint: 'main',
		},
	});

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

	let commandEncoder = device.createCommandEncoder();

	let passEncoder = commandEncoder.beginComputePass();

	Timer.timestamp(passEncoder,"sample-start");
	
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	let numPoints = las.header.numPoints;
	let groups = Math.ceil(numPoints / 128);
	passEncoder.dispatch(groups);

	Timer.timestamp(passEncoder,"sample-end", {print: true});
	
	passEncoder.end();

	device.queue.submit([commandEncoder.finish()]);

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

