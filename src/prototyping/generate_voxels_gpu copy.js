


const cs = `


struct Uniforms {
	numTriangles     : u32;
	gridSize         : u32;
	bbMin            : vec3<f32>;      // offset(16)
	bbMax            : vec3<f32>;      // offset(32)
};


struct Dbg {
	counter : atomic<u32>;
};

struct U32s {
	values : array<u32>;
};

struct F32s {
	values : array<f32>;
};

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<storage, read_write> dbg : Dbg;
[[binding(10), group(0)]] var<storage, read> indices : U32s;
[[binding(11), group(0)]] var<storage, read> positions : F32s;
[[binding(12), group(0)]] var<storage, read> uvs : F32s;
[[binding(13), group(0)]] var<storage, read_write> colors : U32s;

fn loadPosition(vertexIndex : u32) -> vec3<f32> {

	var position = vec3<f32>(
		positions.values[3u * vertexIndex + 0u],
		positions.values[3u * vertexIndex + 1u],
		positions.values[3u * vertexIndex + 2u],
	);

	return position;
};

fn toRGBA(position : vec3<f32>) -> u32 {

	// var R = u32(30.0 * (position.x - uniforms.bbMin.x));
	// var G = u32(30.0 * (position.y - uniforms.bbMin.y));
	// var B = u32(100.0 * (position.z - uniforms.bbMin.z));

	var R = u32(255.0 * position.x);
	var G = u32(255.0 * position.y);
	var B = u32(255.0 * position.z);

	R = clamp(R, 0u, 255u);
	G = clamp(G, 0u, 255u);
	B = clamp(B, 0u, 255u);

	// var rgba = (B << 16u);
	// var rgba = (G << 8u);
	// var rgba = R;
	var rgba = (B << 16u) | (G << 8u) | R;

	return rgba;
}

[[stage(compute), workgroup_size(128)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	var triangleIndex = GlobalInvocationID.x;

	if(triangleIndex >= uniforms.numTriangles){
		return;
	}

	ignore(uniforms);
	var a = positions.values[0];
	var b = uvs.values[0];
	var c = indices.values[0];
	var d = colors.values[0];
	ignore(dbg);


	var i0 = indices.values[3u * triangleIndex + 0u];
	var i1 = indices.values[3u * triangleIndex + 1u];
	var i2 = indices.values[3u * triangleIndex + 2u];

	var p0 = loadPosition(i0);
	var p1 = loadPosition(i1);
	var p2 = loadPosition(i2);

	var center = (p0 + p1 + p2) / 3.0;

	var boxSize = uniforms.bbMax - uniforms.bbMin;
	var gridSize = max(max(boxSize.x, boxSize.y), boxSize.z);
	var gridIndexD = f32(uniforms.gridSize) * (center - uniforms.bbMin) / gridSize;
	var voxelIndex3D = vec3<u32>(gridIndexD);
	var voxelIndex1D = voxelIndex3D.x 
		+ uniforms.gridSize * voxelIndex3D.y
		+ uniforms.gridSize * uniforms.gridSize * voxelIndex3D.z;

	colors.values[triangleIndex] = 123u * voxelIndex1D;


	colors.values[triangleIndex] = toRGBA(p0);



	var u = uvs.values[2u * triangleIndex + 0u];
	var v = uvs.values[2u * triangleIndex + 1u];

	colors.values[i0] = toRGBA(vec3<f32>(f32(i0 / 100000u), 0.0, 0.0));
	// colors.values[triangleIndex] = toRGBA(vec3<f32>(u, v, 0.0));

	// if(indices.values[3] == 2u){
	// 	colors.values[triangleIndex] = 0x0000FF00u;
	// }

	colors.values[triangleIndex] = triangleIndex % 255u;


	return;
}
`;


let gridSize = 32;

export function generateVoxelsGpu(renderer, node){

	let positions = node.geometry.buffers.find(buffer => buffer.name === "position");
	let uvs = node.geometry.buffers.find(buffer => buffer.name === "uv");
	let colors = node.geometry.buffers.find(buffer => buffer.name === "color");

	let vboPositions = renderer.getGpuBuffer(positions.buffer);
	let vboUVs = renderer.getGpuBuffer(uvs.buffer);
	let vboIndices = renderer.getGpuBuffer(node.geometry.indices);
	let vboColors = renderer.getGpuBuffer(colors.buffer);

	let numTriangles = node.geometry.indices.length / 3;

	
	// SETUP
	let {device} = renderer;

	let pipeline = device.createComputePipeline({
		compute: {
			module: device.createShaderModule({code: cs}),
			entryPoint: 'main',
		},
	});


	let storage_flags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
	let uniform_flags = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
	
	let uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});
	let dbgBuffer = device.createBuffer({size: 256, usage: storage_flags});

	let bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: {buffer: dbgBuffer}},
			{binding: 10, resource: {buffer: vboIndices}},
			{binding: 11, resource: {buffer: vboPositions}},
			{binding: 12, resource: {buffer: vboUVs}},
			{binding: 13, resource: {buffer: vboColors}},
		],
	});

	{ // SET UNIFORMS
		let buffer = new ArrayBuffer(256);
		let view = new DataView(buffer);

		view.setUint32(0, numTriangles, true);
		view.setUint32(4, gridSize, true);

		let box = node.boundingBox;
		view.setFloat32(16, box.min.x, true);
		view.setFloat32(20, box.min.y, true);
		view.setFloat32(24, box.min.z, true);

		view.setFloat32(32, box.max.x, true);
		view.setFloat32(36, box.max.y, true);
		view.setFloat32(40, box.max.z, true);

		// view.setFloat32(16, 1, true);
		// view.setFloat32(20, 2, true);
		// view.setFloat32(24, 3, true);

		// view.setFloat32(32, 4, true);
		// view.setFloat32(36, 5, true);
		// view.setFloat32(40, 6, true);

		device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
	}


	// DISPATCH
	const commandEncoder = device.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	let numGroups = Math.floor(numTriangles / 128);
	passEncoder.dispatch(numGroups);

	passEncoder.endPass();
	device.queue.submit([commandEncoder.finish()]);


	// READ RESULTS
	renderer.readBuffer(dbgBuffer, 0, 4).then(result => {
		console.log("dbg: ", new Uint32Array(result)[0]);
	});

}