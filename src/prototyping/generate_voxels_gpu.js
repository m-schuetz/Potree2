
import {SceneNode, Vector3, Box3, SPECTRAL} from "potree";
import {render as renderVoxels} from "./generate_voxels_gpu_render.js";

let gridSize = 64;
let fboSize = 64;

const sourceTriangleProjection = `

struct F32s {
	values : array<f32>;
};

struct U32s {
	values : array<u32>;
};

struct AU32s {
	values : array<atomic<u32>>;
};

struct Uniforms {
	screen_width     : f32;
	screen_height    : f32;
	instance         : u32;
	bbMin            : vec3<f32>;      // offset(16)
	bbMax            : vec3<f32>;      // offset(32)
};

struct Dbg {
	value : u32;
	triangleCount : atomic<u32>;
};

struct VertexInput {
	@builtin(instance_index) instanceIndex : u32;
	@builtin(vertex_index) index : u32,
};

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec4<f32>,
	@location(1) voxelPos : vec3<f32>;
	@location(2) triangleIndex : u32;
	@location(3) objectPos : vec3<f32>;
};

struct FragmentInput {
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec4<f32>,
	@location(1) voxelPos : vec3<f32>;
	@location(2) triangleIndex : u32;
	@location(3) objectPos : vec3<f32>;
};

struct FragmentOutput {
	@location(0) color : vec4<f32>,
};


@binding(0) @group(0) var<uniform> uniforms : Uniforms;

[[binding(10), group(0)]] var<storage, read> indices : U32s;
[[binding(11), group(0)]] var<storage, read> positions : F32s;
[[binding(12), group(0)]] var<storage, read> uvs : F32s;
[[binding(13), group(0)]] var<storage, read> colors : U32s;

[[binding(50), group(0)]] var<storage, read_write> dbg : Dbg;

[[binding(100), group(0)]] var<storage, read_write> voxelGrid : U32s;
[[binding(101), group(0)]] var<storage, read_write> triangleFlagBuffer : AU32s;
[[binding(110), group(0)]] var<storage, read_write> outPositions : F32s;
[[binding(111), group(0)]] var<storage, read_write> outColors : U32s;

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

fn unpackColor(uColor : u32) -> vec4<f32> {

	var R = (uColor >>  0u) & 0xFFu;
	var G = (uColor >>  8u) & 0xFFu;
	var B = (uColor >> 16u) & 0xFFu;

	var color = vec4<f32>(
		f32(R) / 255.0,
		f32(G) / 255.0,
		f32(B) / 255.0,
		255.0,
	);

	return color;
}

fn loadPosition(vertexIndex : u32) -> vec3<f32> {
	
	var position = vec3<f32>(
		positions.values[3u * vertexIndex + 0u],
		positions.values[3u * vertexIndex + 1u],
		positions.values[3u * vertexIndex + 2u],
	);

	return position;
};

// fn loadColor(vertexIndex : u32) -> vec3<f32> {

// 	var rgba = colors.values[vertexIndex];
	
// 	return unpackColor(rgba);
// };



@stage(vertex)
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

	var length_01 = 5.0 * length(p0 - p1);
	var length_02 = 5.0 * length(p0 - p2);

	var n_length_01 = max(length_01, 4.0 / ${fboSize}.0);
	var n_length_02 = max(length_02, 4.0 / ${fboSize}.0);

	var vertexPosition : vec3<f32>;
	var localIndex = vertex.index % 3u;
	if(localIndex == 0u){
		result.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
		vertexPosition = p0;
	}else if(localIndex == 1u){
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

	result.triangleIndex = triangleIndex;
	result.objectPos = vertexPosition;

	return result;
}


@stage(fragment)
fn main_fragment(fragment : FragmentInput) -> FragmentOutput {

	{
		var f240t = outPositions.values[0];
		var f2g4q = outColors.values[0];
	}

	var result : FragmentOutput;
	result.color = fragment.color;

	dbg.value = u32(3000.0 * fragment.color.x);

	var voxelPos = vec3<u32>(fragment.voxelPos);
	var voxelIndex = toIndex1D(${gridSize}u, voxelPos);

	var outsideX = fragment.voxelPos.x < 0.0 || fragment.voxelPos.x > ${gridSize}.0;
	var outsideY = fragment.voxelPos.y < 0.0 || fragment.voxelPos.y > ${gridSize}.0;
	var outsideZ = fragment.voxelPos.z < 0.0 || fragment.voxelPos.z > ${gridSize}.0;
	if(outsideX || outsideY || outsideZ){
		// voxelGrid.values[voxelIndex] = packColor(vec4<f32>(1.0, 0.0, 0.0, 1.0));
	}else{
		var oldCount = atomicAdd(&triangleFlagBuffer.values[fragment.triangleIndex], 1u);
		if(oldCount == 0u){
			var oldTriangleCount = atomicAdd(&dbg.triangleCount, 1u);

			// outPositions.values[3u * oldTriangleCount + 0u] = fragment.objectPos.x;
			// outPositions.values[3u * oldTriangleCount + 1u] = fragment.objectPos.y;
			// outPositions.values[3u * oldTriangleCount + 2u] = fragment.objectPos.z;

			// outColors.values[oldTriangleCount] = packColor(fragment.color);

			var i0 = indices.values[3u * fragment.triangleIndex + 0u];
			var i1 = indices.values[3u * fragment.triangleIndex + 1u];
			var i2 = indices.values[3u * fragment.triangleIndex + 2u];

			var p0 = loadPosition(i0);
			var p1 = loadPosition(i1);
			var p2 = loadPosition(i2);

			outPositions.values[9u * oldTriangleCount + 0u] = p0.x;
			outPositions.values[9u * oldTriangleCount + 1u] = p0.y;
			outPositions.values[9u * oldTriangleCount + 2u] = p0.z;
			outPositions.values[9u * oldTriangleCount + 3u] = p1.x;
			outPositions.values[9u * oldTriangleCount + 4u] = p1.y;
			outPositions.values[9u * oldTriangleCount + 5u] = p1.z;
			outPositions.values[9u * oldTriangleCount + 6u] = p2.x;
			outPositions.values[9u * oldTriangleCount + 7u] = p2.y;
			outPositions.values[9u * oldTriangleCount + 8u] = p2.z;

			outColors.values[3u * oldTriangleCount + 0u] = colors.values[i0];
			outColors.values[3u * oldTriangleCount + 1u] = colors.values[i1];
			outColors.values[3u * oldTriangleCount + 2u] = colors.values[i2];

			// outColors.values[oldTriangleCount] = packColor(fragment.color);

			
		}

		voxelGrid.values[voxelIndex] = packColor(fragment.color);

		result.color = vec4<f32>(
			fragment.position.x / ${fboSize}.0, 
			fragment.position.y / ${fboSize}.0, 
			0.0, 
			1.0,
		);
	}

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


let initialized = false;
let pipeline = null;
let bindGroup = null;
let fbo = null;
let uniformBuffer = null;
let gridBuffer = null;
let gridResetBuffer = null;
let triangleFlagBuffer = null;
let dbgBuffer = null;
let outPositionsBuffer = null;
let outColorsBuffer = null;

function init(renderer, node){

	let {device} = renderer;

	fbo = renderer.getFramebuffer("voxel_target");
	fbo.setSize(fboSize, fboSize);

	pipeline = device.createRenderPipeline({
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
	
	uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});
	dbgBuffer = device.createBuffer({size: 256, usage: storage_flags});
	
	let numTriangles = node.geometry.indices.length / 3;
	gridResetBuffer = new Uint8Array(4 * (gridSize) ** 3);
	gridBuffer = device.createBuffer({size: gridResetBuffer.byteLength, usage: storage_flags});
	triangleFlagBuffer = device.createBuffer({size: 4 * numTriangles, usage: storage_flags});

	outPositionsBuffer = device.createBuffer({size: 128_000_000, usage: storage_flags});
	outColorsBuffer = device.createBuffer({size: 128_000_000, usage: storage_flags});

	let indexBuffer = renderer.getGpuBuffer(node.geometry.indices);
	let vbos = renderer.getGpuBuffers(node.geometry);
	let vboPosition = vbos.find(item => item.name === "position").vbo;
	let vboUV = vbos.find(item => item.name === "uv").vbo;
	let vboColor = vbos.find(item => item.name === "color").vbo;

	bindGroup = renderer.device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding:   0, resource: {buffer: uniformBuffer}},
			{binding:  10, resource: {buffer: indexBuffer}},
			{binding:  11, resource: {buffer: vboPosition}},
			{binding:  12, resource: {buffer: vboUV}},
			{binding:  13, resource: {buffer: vboColor}},
			{binding:  50, resource: {buffer: dbgBuffer}},
			{binding: 100, resource: {buffer: gridBuffer}},
			{binding: 101, resource: {buffer: triangleFlagBuffer}},
			{binding: 110, resource: {buffer: outPositionsBuffer}},
			{binding: 111, resource: {buffer: outColorsBuffer}},
		]
	});

	window.fbo = fbo;
}

async function voxelize(renderer, node, boundingBox, chunkName){
	let {device} = renderer;

	let box = boundingBox.clone();
	let min = box.min.clone();
	let cube = box.cube();
	let cubeSize = cube.size().x;
	box = cube;

	let renderPassDescriptor = {
		colorAttachments: [{
			view: fbo.colorAttachments[0].texture.createView(), 
			loadOp: "clear", clearValue: { r: 0.5, g: 0, b: 0, a: 1.0 },
			storeOp: 'store',
		}],
		depthStencilAttachment: {
			view: fbo.depth.texture.createView(),
			depthLoadOp: "load",
			depthStoreOp: "store",
		},
		sampleCount: 1,
	};

	{ // SET UNIFORMS
		let buffer = new ArrayBuffer(256);
		let view = new DataView(buffer);

		// view.setUint32( 8, Math.random() * 10, true);

		view.setFloat32(16, box.min.x, true);
		view.setFloat32(20, box.min.y, true);
		view.setFloat32(24, box.min.z, true);

		view.setFloat32(32, box.max.x, true);
		view.setFloat32(36, box.max.y, true);
		view.setFloat32(40, box.max.z, true);

		device.queue.writeBuffer(uniformBuffer, 0, buffer, 0, buffer.byteLength);
	}

	let numTriangles = node.geometry.indices.length / 3;

	renderer.fillBuffer(dbgBuffer, 0, 2);
	renderer.fillBuffer(gridBuffer, 0, gridSize ** 3);
	renderer.fillBuffer(triangleFlagBuffer, 0, numTriangles);
	

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	passEncoder.draw(node.geometry.indices.length, 1, 0, 0);

	passEncoder.end();
	let commandBuffer = commandEncoder.finish();
	renderer.device.queue.submit([commandBuffer]);

	// READ RESULTS
	let voxels = {
		positions: null,
		colors: null,
		numVoxels: 0,
	};


	let maxAccept = 1_000;
	let p1 = renderer.readBuffer(gridBuffer, 0, 4 * (gridSize) ** 3);
	// let p2 = renderer.readBuffer(dbgBuffer, 0, 8);
	// let p3 = renderer.readBuffer(outPositionsBuffer, 0, maxAccept * 3 * 12);
	// let p4 = renderer.readBuffer(outColorsBuffer, 0, maxAccept * 3 * 4);

	let result = await p1;
	// let numAcceptedTriangles = new Uint32Array(await p2)[1];
	

	let triangles = null;
	// if(numAcceptedTriangles < maxAccept){

	// 	let outPositions = new Float32Array(await p3);
	// 	let outColors = new Uint32Array(await p4);

	// 	let clippedPositions = new Float32Array(outPositions.buffer, 0, 9 * numAcceptedTriangles);
	// 	let clippedColors = new Float32Array(outColors.buffer, 0, 3 * numAcceptedTriangles);

	// 	// potree.onUpdate(() => {
	// 	// 	potree.renderer.drawMesh({positions: clippedPositions, colors: clippedColors});
	// 	// 	// potree.renderer.drawQuads(clippedPositions, clippedColors);
	// 	// });

	// 	triangles = {
	// 		positions: clippedPositions,
	// 		colors: clippedColors,
	// 	};

	// }

	// let result = await renderer.readBuffer(gridBuffer, 0, 4 * (gridSize) ** 3);
	// let numAcceptedTriangles = new Uint32Array(await renderer.readBuffer(dbgBuffer, 0, 8))[1];

	{

		let numVoxels = 0;
		let u32 = new Uint32Array(result);

		for(let i = 0; i < u32.length; i++){
			if(u32[i] > 0){
				numVoxels++;
			}
		}

		let positions = new Float32Array(3 * numVoxels);
		let colors = new Uint8Array(4 * numVoxels);

		let cellSize = cubeSize / gridSize;
		let numAdded = 0;
		for(let i = 0; i < u32.length; i++){
			if(u32[i] > 0){

				numAdded++;

				let [ix, iy, iz] = toIndex3D(gridSize, i);
				// let color = unpackColor(u32[i]);

				let x = cubeSize * (ix / gridSize) + min.x + cellSize / 2;
				let y = cubeSize * (iy / gridSize) + min.y + cellSize / 2;
				let z = cubeSize * (iz / gridSize) + min.z + cellSize / 2;

				positions[3 * numAdded + 0] = x;
				positions[3 * numAdded + 1] = y;
				positions[3 * numAdded + 2] = z;

				let rgba = u32[i];
				colors[4 * numAdded + 0] = (rgba >>>  0) & 0xFF;
				colors[4 * numAdded + 1] = (rgba >>>  8) & 0xFF;
				colors[4 * numAdded + 2] = (rgba >>> 16) & 0xFF;
				colors[4 * numAdded + 3] = 255;
			}
		}

		voxels.positions = positions;
		voxels.colors = colors;
		voxels.numVoxels = numVoxels;
	}

	// if(voxels.numVoxels > 0){
	// 	console.log(`node(${chunkName}), numVoxels(${voxels.numVoxels}, numAcceptedTriangles(${numAcceptedTriangles})`);
	// }

	let chunk = {
		boundingBox: box,
		voxels,
		triangles: triangles,
	};

	return chunk;
}

class Chunk{

	constructor(){
		this.boundingBox = new Box3();
		this.voxels = [];
		this.triangles = null;
		this.children = [];
		this.level = 0;
		this.parent = null;
		this.visible = true;
		this.name = "";
	}

	expand(){

		let childSize = this.boundingBox.size().divideScalar(2);
		let parentMin = this.boundingBox.min;

		for(let i = 0; i < 8; i++){

			let x = (i >> 0) & 1;
			let y = (i >> 1) & 1;
			let z = (i >> 2) & 1;

			let min = new Vector3(
				parentMin.x + x * childSize.x,
				parentMin.y + y * childSize.y,
				parentMin.z + z * childSize.z,
			);
			let max = min.clone().add(childSize);

			let childBox = new Box3(min, max);

			let child = new Chunk();
			child.boundingBox = childBox;
			child.level = this.level + 1;
			child.parent = this;
			child.name = this.name + i;
			
			this.children.push(child);
		}
	}

	traverse(callback, level = 0){

		callback(this, level);

		for(let child of this.children){
			child.traverse(callback, level + 1);
		}

	}

	

};

class VoxelTree extends SceneNode{

	constructor(){
		super("abc");

		this.root = new Chunk();
		this.root.name = "r";
		this.gridSize = gridSize;
	}

	getRootVoxelSize(){
		let box = this.root.boundingBox;
		let cubeSize = box.size().x;

		let voxelSize = cubeSize / this.gridSize;

		return voxelSize;
	}

	render(drawstate){
		renderVoxels(this, drawstate);
		// console.log("asdf");
	}

	traverse(callback){
		this.root.traverse(callback, 0);
	}

	traverseBreadthFirst(callback){

		let stack = [this.root];

		while(stack.length > 0){
			let node = stack.shift();

			callback(node);

			for(let child of node.children){

				if(child.visible){
					stack.push(child);
				}
			}
		}

	}

};



export async function generateVoxelsGpu(renderer, node){

	init(renderer, node);

	console.time("generating voxels");

	let box = node.boundingBox.clone();
	// box.min.set(1.2, -0.4, -4.8);
	// box.max.set(1.5, 0.3, -4.1);


	let cube = box.cube();
	let center = cube.center();

	let root = new Chunk();
	root.name = "r";
	root.boundingBox = cube.clone();
	root.expand();
	window.root = root;
	for(let child of root.children){
		child.expand();

		for(let child1 of child.children){
			child1.expand();

			// for(let child2 of child1.children){
			// 	child2.expand();
			// }
		}
	}

	let instance = 0;
	let promises = [];
	root.traverse( async (chunk) => {
		let promise = voxelize(renderer, node, chunk.boundingBox, chunk.name);
		promises.push(promise);

		instance++;

		promise.then(result => {
			let voxels = result.voxels;
			chunk.voxels = voxels;
			chunk.triangles = result.triangles;
		});

	});

	await Promise.all(promises);

	console.timeEnd("generating voxels");

	let voxelTree = new VoxelTree();
	voxelTree.root = root;
	scene.root.children.push(voxelTree);

	potree.onUpdate( () => {
			
		// potree.renderer.drawBoundingBox(
		// 	box.center(),
		// 	box.size(),
		// 	new Vector3(255, 255, 0),
		// );

		// potree.renderer.drawBoundingBox(
		// 	cube.center(),
		// 	cube.size(),
		// 	new Vector3(255, 0, 0),
		// );

		root.traverse( (chunk, level) => {

			chunk.visible = false;

			if(chunk.voxels.numVoxels === 0){
				return;
			}

			// let whitelist = ["r", "r4", "r42"];
			// if(!whitelist.includes(chunk.name)){
			// 	return;
			// }

			let color = new Vector3(...SPECTRAL.get(level / 4));

			let center = chunk.boundingBox.center();
			let size = chunk.boundingBox.size().length();
			let camWorldPos = camera.getWorldPosition();
			let distance = camWorldPos.distanceTo(center);

			let expand = (size / distance) > 0.8;

			chunk.visible = expand;

			if(chunk.visible && chunk.triangles){
				potree.renderer.drawMesh(chunk.triangles);
			}

			if(expand){
				potree.renderer.drawBoundingBox(
					chunk.boundingBox.center(),
					chunk.boundingBox.size(),
					color,
				);
			}

		});


	});


}
