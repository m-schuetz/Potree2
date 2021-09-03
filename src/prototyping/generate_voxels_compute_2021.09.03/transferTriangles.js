
import {storage_flags, uniform_flags} from "./common.js";

let csTransfer = `

[[block]] struct Uniforms {
	batchTriangleCount      : u32;
	fragmentTriangleCount   : u32;
	firstTriangle           : u32;
	chunkGridSize           : u32;
	numTransfered           : u32;
};

[[block]] struct F32s { values : [[stride(4)]] array<f32>; };
[[block]] struct U32s { values : [[stride(4)]] array<u32>; };
[[block]] struct I32s { values : [[stride(4)]] array<i32>; };
[[block]] struct AU32s { values : [[stride(4)]] array<atomic<u32>>; };
[[block]] struct AI32s { values : [[stride(4)]] array<atomic<i32>>; };

[[binding( 0), group(0)]] var<uniform> uniforms : Uniforms;

[[binding(10), group(0)]] var<storage, read_write> sortedTriangles : U32s;

[[binding(20), group(0)]] var<storage, read_write> positions : F32s;
[[binding(21), group(0)]] var<storage, read_write> colors    : U32s;


fn getNumChunkGridCells() -> u32 {
	var chunkGridSize = uniforms.chunkGridSize;
	var numChunkGridCells = chunkGridSize * chunkGridSize * chunkGridSize;

	return numChunkGridCells;
};

fn doIgnore(){
	
	ignore(uniforms);
	ignore(sortedTriangles);
	ignore(positions);
	ignore(colors);

}


[[stage(compute), workgroup_size(128)]]
fn main_transfer_triangles([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	doIgnore();

	if(GlobalInvocationID.x >= uniforms.fragmentTriangleCount){
		return;
	}

	var triangleIndex = uniforms.firstTriangle + GlobalInvocationID.x;
	var outTriangleIndex = GlobalInvocationID.x + uniforms.numTransfered;
	
	var offset_pos = 0u;
	var offset_color = 9u * uniforms.batchTriangleCount;

	var trianglePosOffset = offset_pos + 9u * triangleIndex;
	var x0 = bitcast<f32>(sortedTriangles.values[trianglePosOffset + 0u]);
	var y0 = bitcast<f32>(sortedTriangles.values[trianglePosOffset + 1u]);
	var z0 = bitcast<f32>(sortedTriangles.values[trianglePosOffset + 2u]);
	var x1 = bitcast<f32>(sortedTriangles.values[trianglePosOffset + 3u]);
	var y1 = bitcast<f32>(sortedTriangles.values[trianglePosOffset + 4u]);
	var z1 = bitcast<f32>(sortedTriangles.values[trianglePosOffset + 5u]);
	var x2 = bitcast<f32>(sortedTriangles.values[trianglePosOffset + 6u]);
	var y2 = bitcast<f32>(sortedTriangles.values[trianglePosOffset + 7u]);
	var z2 = bitcast<f32>(sortedTriangles.values[trianglePosOffset + 8u]);

	positions.values[9u * outTriangleIndex + 0u] = x0;
	positions.values[9u * outTriangleIndex + 1u] = y0;
	positions.values[9u * outTriangleIndex + 2u] = z0;
	positions.values[9u * outTriangleIndex + 3u] = x1;
	positions.values[9u * outTriangleIndex + 4u] = y1;
	positions.values[9u * outTriangleIndex + 5u] = z1;
	positions.values[9u * outTriangleIndex + 6u] = x2;
	positions.values[9u * outTriangleIndex + 7u] = y2;
	positions.values[9u * outTriangleIndex + 8u] = z2;

	var triangleColorOffset = offset_color + 3u * triangleIndex;
	var c0 = sortedTriangles.values[triangleColorOffset + 0u];
	var c1 = sortedTriangles.values[triangleColorOffset + 1u];
	var c2 = sortedTriangles.values[triangleColorOffset + 2u];

	colors.values[3u * outTriangleIndex + 0u] = c0;
	colors.values[3u * outTriangleIndex + 1u] = c1;
	colors.values[3u * outTriangleIndex + 2u] = c2;

}

`;

export function transferTriangles(renderer, batch, chunk, chunkGridSize, numTriangles, firstTriangle, numTransfered){

	let {device} = renderer;

	let uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});
	{
		let buffer = new ArrayBuffer(256);
		let view = new DataView(buffer);

		view.setUint32( 0, batch.numTriangles, true);
		view.setUint32( 4, numTriangles, true);
		view.setUint32( 8, firstTriangle, true);
		view.setUint32(12, chunkGridSize, true);
		view.setUint32(16, numTransfered, true);

		device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
	}

	let bindGroups = [{
		location: 0,
		entries: [
			{binding:  0, resource: {buffer: uniformBuffer}},

			{binding: 10, resource: {buffer: batch.gpu_sortedTriangles}},

			{binding: 20, resource: {buffer: chunk.gpu_positions}},
			{binding: 21, resource: {buffer: chunk.gpu_colors}},
		],
	}];

	renderer.runCompute({
		code: csTransfer,
		entryPoint: "main_transfer_triangles",
		bindGroups: bindGroups,
		dispatchGroups: [Math.ceil(numTriangles / 128)],
	});

}