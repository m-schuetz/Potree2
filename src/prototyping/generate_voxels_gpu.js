
import {Vector3, Box3} from "potree";

let gridSize = 64;
let fboSize = 64;

const sourceTriangleProjection = `

[[block]] struct F32s {
	values : [[stride(4)]] array<f32>;
};

[[block]] struct U32s {
	values : [[stride(4)]] array<u32>;
};

[[block]] struct Uniforms {
	screen_width     : f32;
	screen_height    : f32;
	bbMin            : vec3<f32>;      // offset(16)
	bbMax            : vec3<f32>;      // offset(32)
};

[[block]] struct Dbg {
	value : u32;
};

struct VertexInput {
	[[builtin(instance_index)]] instanceIndex : u32;
	[[builtin(vertex_index)]] index : u32;
};

struct VertexOutput {
	[[builtin(position)]] position : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
	[[location(1)]] voxelPos : vec3<f32>;
};

struct FragmentInput {
	[[builtin(position)]] position : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
	[[location(1)]] voxelPos : vec3<f32>;
};

struct FragmentOutput {
	[[location(0)]] color : vec4<f32>;
};


[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;

[[binding(10), group(0)]] var<storage, read> indices : U32s;
[[binding(11), group(0)]] var<storage, read> positions : F32s;
[[binding(12), group(0)]] var<storage, read> uvs : F32s;
[[binding(13), group(0)]] var<storage, read> colors : U32s;

[[binding(50), group(0)]] var<storage, read_write> dbg : Dbg;

[[binding(100), group(0)]] var<storage, read_write> voxelGrid : U32s;

fn loadPosition(vertexIndex : u32) -> vec3<f32> {
	
	var position = vec3<f32>(
		positions.values[3u * vertexIndex + 0u],
		positions.values[3u * vertexIndex + 1u],
		positions.values[3u * vertexIndex + 2u],
	);

	return position;
};

fn toIndex1D(gridSize : u32, voxelPos : vec3<u32>) -> u32{
	return voxelPos.x 
		+ gridSize * voxelPos.y 
		+ gridSize * gridSize * voxelPos.z;
}

fn packColor(color : vec4<f32>) -> u32 {

	var R = u32(255.0 * color.r);
	var G = u32(255.0 * color.g);
	var B = u32(255.0 * color.b);

	var packed = 
		  (R <<  0u)
		| (G <<  8u)
		| (B << 16u);

	return packed;
}

[[stage(vertex)]]
fn main_vertex(vertex : VertexInput) -> VertexOutput {

	var result : VertexOutput;
	result.color = vec4<f32>(0.0, 1.0, 0.0, 1.0);

	{
		var a = indices.values[0];
		var b = positions.values[0];
		var c = uvs.values[0];
		var d = colors.values[0];
	}

	var triangleIndex = vertex.index / 3u;
	var i0 = indices.values[3u * triangleIndex + 0u];
	var i1 = indices.values[3u * triangleIndex + 1u];
	var i2 = indices.values[3u * triangleIndex + 2u];

	var uColor = colors.values[i0];
	var R = (uColor >>  0u) & 0xFFu;
	var G = (uColor >>  8u) & 0xFFu;
	var B = (uColor >> 16u) & 0xFFu;
	var color = vec4<f32>(f32(R), f32(G), f32(B), 255.0) / 255.0;

	var p0 = loadPosition(i0);
	var p1 = loadPosition(i1);
	var p2 = loadPosition(i2);

	var length_01 = 20.0 * length(p0 - p1);
	var length_02 = 20.0 * length(p0 - p2);

	var n_length_01 = max(length_01, 1.0 / 1024.0);
	var n_length_02 = max(length_02, 1.0 / 1024.0);

	var vertexPosition : vec3<f32>;
	var localIndex = vertex.index % 3u;
	if(localIndex == 0u){
		result.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
		vertexPosition = p0;
	}elseif(localIndex == 1u){
		result.position = vec4<f32>(n_length_01 - 1.0, -1.0, 0.0, 1.0);
		vertexPosition = p1;
	}if(localIndex == 2u){
		result.position = vec4<f32>(-1.0, n_length_02 - 1.0, 0.0, 1.0);
		vertexPosition = p2;
	}

	result.color = color;

	{
		var bbMin = vec3<f32>(uniforms.bbMin.x, uniforms.bbMin.y, uniforms.bbMin.z);
		var bbMax = vec3<f32>(uniforms.bbMax.x, uniforms.bbMax.y, uniforms.bbMax.z);
		var bbSize = bbMax - bbMin;
		var cubeSize = max(max(bbSize.x, bbSize.y), bbSize.z);
		var gridSize = ${gridSize}.0;

		var gx = gridSize * (vertexPosition.x - uniforms.bbMin.x) / cubeSize;
		var gy = gridSize * (vertexPosition.y - uniforms.bbMin.y) / cubeSize;
		var gz = gridSize * (vertexPosition.z - uniforms.bbMin.z) / cubeSize;

		result.voxelPos = vec3<f32>(gx, gy, gz);
	}

	return result;
}


[[stage(fragment)]]
fn main_fragment(fragment : FragmentInput) -> FragmentOutput {

	var result : FragmentOutput;
	result.color = fragment.color;

	dbg.value = u32(3000.0 * fragment.color.x);
	voxelGrid.values[0] = 456u;
	voxelGrid.values[1] = 789u;

	var voxelPos = vec3<u32>(fragment.voxelPos);
	var voxelIndex = toIndex1D(${gridSize}u, voxelPos);

	// voxelGrid.values[voxelIndex] = 1u;
	voxelGrid.values[voxelIndex] = packColor(fragment.color);

	result.color = vec4<f32>(
		fragment.position.x / ${fboSize}.0, 
		fragment.position.y / ${fboSize}.0, 
		0.0, 
		1.0,
	);

	return result;
}

`;




function toIndex1D(gridSize, x, y, z){
	return x + gridSize * y + gridSize * gridSize * z
}

function toIndex3D(gridSize, index){
	let z = Math.floor(index / (gridSize * gridSize));
	let y = Math.floor((index - gridSize * gridSize * z) / gridSize);
	let x = index % gridSize;

	return [x, y, z];
}

function unpackColor(uColor){
	let R = (uColor >>>  0) & 0xFF;
	let G = (uColor >>>  8) & 0xFF;
	let B = (uColor >>> 16) & 0xFF;

	let color = new Vector3(R, G, B);

	return color;
}

export function generateVoxelsGpu(renderer, node){

	let {device} = renderer;

	console.time("generating voxels");

	let box = node.boundingBox;
	let min = box.min.clone();
	let cubeSize = Math.max(...box.size().toArray());
	let cube = new Box3(min, min.clone().addScalar(cubeSize));

	let fbo = renderer.getFramebuffer("voxel_target");
	fbo.setSize(fboSize, fboSize);

	let pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: sourceTriangleProjection}),
			entryPoint: "main_vertex",
			buffers: [],
		},
		fragment: {
			module: device.createShaderModule({code: sourceTriangleProjection}),
			entryPoint: "main_fragment",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: false,
			depthCompare: "always",
			format: "depth32float",
		},
	});

	let storage_flags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
	let uniform_flags = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
	
	let uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});
	let dbgBuffer = device.createBuffer({size: 256, usage: storage_flags});
	let gridBuffer = device.createBuffer({size: 4 * (gridSize) ** 3, usage: storage_flags});

	let indexBuffer = renderer.getGpuBuffer(node.geometry.indices);
	let vbos = renderer.getGpuBuffers(node.geometry);
	let vboPosition = vbos.find(item => item.name === "position").vbo;
	let vboUV = vbos.find(item => item.name === "uv").vbo;
	let vboColor = vbos.find(item => item.name === "color").vbo;

	let bindGroup = renderer.device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 10, resource: {buffer: indexBuffer}},
			{binding: 11, resource: {buffer: vboPosition}},
			{binding: 12, resource: {buffer: vboUV}},
			{binding: 13, resource: {buffer: vboColor}},
			{binding: 50, resource: {buffer: dbgBuffer}},
			{binding: 100, resource: {buffer: gridBuffer}},
		]
	});

	let renderPassDescriptor = {
		colorAttachments: [{
			view: fbo.colorAttachments[0].texture.createView(), 
			loadValue: { r: 0.5, g: 0, b: 0, a: 1.0 }
		}],
		depthStencilAttachment: {
			view: fbo.depth.texture.createView(),
			depthLoadValue: "load",
			depthStoreOp: "store",
			stencilLoadValue: 0,
			stencilStoreOp: "store",
		},
		sampleCount: 1,
	};

	{ // SET UNIFORMS
		let buffer = new ArrayBuffer(256);
		let view = new DataView(buffer);

		// view.setUint32(0, numTriangles, true);
		// view.setUint32(4, gridSize, true);

		// let box = node.boundingBox;
		view.setFloat32(16, box.min.x, true);
		view.setFloat32(20, box.min.y, true);
		view.setFloat32(24, box.min.z, true);

		view.setFloat32(32, box.max.x, true);
		view.setFloat32(36, box.max.y, true);
		view.setFloat32(40, box.max.z, true);

		device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
	}

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);


	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	// passEncoder.setVertexBuffer(0, vboPosition);
	// passEncoder.setVertexBuffer(1, vboUV);
	// passEncoder.setVertexBuffer(2, vboColor);

	// passEncoder.setIndexBuffer(indexBuffer, "uint32", 0, indexBuffer.byteLength);

	let numTriangles = node.geometry.indices.length / 3;
	let numVertices = 3;
	// let numVertices = 900_000;
	passEncoder.draw(node.geometry.indices.length, 1, 0, 0);


	passEncoder.endPass();
	let commandBuffer = commandEncoder.finish();
	renderer.device.queue.submit([commandBuffer]);

	window.fbo = fbo;

	// READ RESULTS
	renderer.readBuffer(gridBuffer, 0, 4 * (gridSize) ** 3).then(result => {
		let u32 = new Uint32Array(result);

		let voxels = [];
		let numCells = 0;
		let cellSize = cubeSize / gridSize;
		for(let i = 0; i < u32.length; i++){
			if(u32[i] > 0){
				numCells++;

				let [ix, iy, iz] = toIndex3D(gridSize, i);
				let color = unpackColor(u32[i]);

				let x = cubeSize * (ix / gridSize) + min.x + cellSize / 2;
				let y = cubeSize * (iy / gridSize) + min.y + cellSize / 2;
				let z = cubeSize * (iz / gridSize) + min.z + cellSize / 2;

				let position = new Vector3(x, y, z);
				let size = new Vector3(cellSize, cellSize, cellSize);

				let voxel = {position, size, color};

				voxels.push(voxel);
			}
		}

		console.log("numCells: ", numCells);

		console.timeEnd("generating voxels");

		potree.onUpdate( () => {
			

			potree.renderer.drawBoundingBox(
				box.center(),
				box.size(),
				new Vector3(255, 255, 0),
			);

			potree.renderer.drawBoundingBox(
				cube.center(),
				cube.size(),
				new Vector3(255, 0, 0),
			);

			for(let voxel of voxels){
				potree.renderer.drawBox(voxel.position, voxel.size, voxel.color);
			}
			// potree.renderer.drawBox(
			// 	new Vector3(0, 0, 0),
			// 	new Vector3(3, 3, 3),
			// 	new Vector3(255, 255, 0),
			// );
		});
	});
	// renderer.readBuffer(dbgBuffer, 0, 4).then(result => {
	// 	console.log("dbg: ", new Uint32Array(result)[0]);
	// });

	

}
