
import {SceneNode, Vector3, Vector4, Matrix4, Box3, Frustum, EventDispatcher, StationaryControls} from "potree";

let shaderCode = `

struct Uniforms {
	worldView        : mat4x4f,
	proj             : mat4x4f,
	screen_width     : f32,
	screen_height    : f32,
	size             : f32,
	elementCounter   : u32,
	hoveredIndex     : i32,
};

struct Node{
	worldView       : mat4x4f,
	ptr_indexBuffer : u32,
	ptr_posBuffer   : u32,
	ptr_uvBuffer    : u32,
	index           : u32,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage> nodes : array<Node>;

@group(1) @binding(0) var<storage> buffer : array<u32>;
@group(1) @binding(1) var _texture        : texture_2d<f32>;
@group(1) @binding(2) var _sampler        : sampler;





fn readU8(offset : u32) -> u32{
	var ipos    = offset / 4u;
	var val_u32 = buffer[ipos];
	var shift   = 8u * (offset % 4u);

	var val_u8  = (val_u32 >> shift) & 0xFFu;

	return val_u8;
}

fn readU16(offset : u32) -> u32{
	
	var first = readU8(offset + 0u);
	var second = readU8(offset + 1u);

	var value = first | (second << 8u);

	return value;
}

fn readI16(offset : u32) -> i32{
	
	var first = u32(readU8(offset + 0u));
	var second = u32(readU8(offset + 1u));

	var sign = second >> 7u;
	second = second & 127u;

	var value = -2;

	if(sign == 0u){
		value = 0;
	}else{
		value = -1;
	}

	var mask = 0xffff0000u;
	value = value | i32(first << 0u);
	value = value | i32(second << 8u);

	return value;
}

fn readU32(offset : u32) -> u32{
	
	var d0 = readU8(offset + 0u);
	var d1 = readU8(offset + 1u);
	var d2 = readU8(offset + 2u);
	var d3 = readU8(offset + 3u);

	var value = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	return value;
}

fn readI32(offset : u32) -> i32{
	
	var d0 = readU8(offset + 0u);
	var d1 = readU8(offset + 1u);
	var d2 = readU8(offset + 2u);
	var d3 = readU8(offset + 3u);

	var value = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	return i32(value);
}

fn readF32(offset : u32) -> f32{
	
	var d0 = readU8(offset + 0u);
	var d1 = readU8(offset + 1u);
	var d2 = readU8(offset + 2u);
	var d3 = readU8(offset + 3u);

	var value_u32 = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	var value_f32 = bitcast<f32>(value_u32);

	return value_f32;
}



struct VertexIn{
	@builtin(vertex_index) vertex_index : u32,
	@builtin(instance_index) instance_index : u32,
};

struct VertexOut{
	@builtin(position) position : vec4<f32>,
	@location(0) @interpolate(flat) pointID : u32,
	@location(1) @interpolate(linear) color : vec4<f32>,
	@location(2) @interpolate(linear) uv : vec2f,
	@location(3) @interpolate(flat)  instanceID : u32,
};

struct FragmentIn{
	@location(0) @interpolate(flat) pointID : u32,
	@location(1) @interpolate(linear) color : vec4<f32>,
	@location(2) @interpolate(linear) uv : vec2f,
	@location(3) @interpolate(flat)  instanceID : u32,
};

struct FragmentOut{
	@location(0) color : vec4<f32>,
	@location(1) point_id : u32,
};

@vertex
fn main_vertex(vertex : VertexIn) -> VertexOut {

	var node = nodes[vertex.instance_index];

	var vertexIndex = readU16(node.ptr_indexBuffer + 2u * vertex.vertex_index);
	var triangleIndex = vertex.vertex_index / 3u;

	var pos = vec4f(
		readF32(node.ptr_posBuffer + 12u * vertexIndex + 0u),
		-readF32(node.ptr_posBuffer + 12u * vertexIndex + 8u),
		readF32(node.ptr_posBuffer + 12u * vertexIndex + 4u),
		1.0,
	);

	var uv = vec2f(
		readF32(node.ptr_uvBuffer + 8u * vertexIndex + 0u),
		readF32(node.ptr_uvBuffer + 8u * vertexIndex + 4u),
	);

	var vout = VertexOut();
	vout.position = uniforms.proj * node.worldView * pos;
	vout.pointID = vertex.instance_index;
	vout.uv = uv;
	vout.instanceID = vertex.instance_index;

	// if(triangleIndex > 1000u){
	// 	vout.instanceID = vout.instanceID + 1u;
	// }

	return vout;
}

@fragment
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout = FragmentOut();
	fout.point_id = uniforms.elementCounter + fragment.pointID;
	// fout.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
	// fout.color = vec4f(fragment.uv.x, fragment.uv.y, 0.0, 1.0);
	// fout.color = vec4f(1.0, 1.0, 1.0, 1.0);

	const SPECTRAL = array(
		vec3f(158.0,   1.0,  66.0),
		vec3f(213.0,  62.0,  79.0),
		vec3f(244.0, 109.0,  67.0),
		vec3f(253.0, 174.0,  97.0),
		vec3f(254.0, 224.0, 139.0),
		vec3f(255.0, 255.0, 191.0),
		vec3f(230.0, 245.0, 152.0),
		vec3f(171.0, 221.0, 164.0),
		vec3f(102.0, 194.0, 165.0),
		vec3f( 50.0, 136.0, 189.0),
		vec3f( 94.0,  79.0, 162.0),
	);

	var node = nodes[fragment.instanceID];

	// fout.color.r = f32(node.index % 10u) / 10.0;

	var color = SPECTRAL[node.index % 10u];
	fout.color.r = color.r / 256.0;
	fout.color.g = color.g / 256.0;
	fout.color.b = color.b / 256.0;

	fout.color = vec4f(1.0, 1.0, 1.0, 1.0);

	fout.color = vec4f(
		fragment.uv.x, 
		fragment.uv.y,
		0.0, 1.0
	);

	var c = textureSample(_texture, _sampler, fragment.uv);
	//_ = _texture;

	fout.color = c;

	return fout;
}

`;

const WGSL_NODE_BYTESIZE = 80;
let initialized = false;
let pipeline = null;
let uniformsBuffer = new ArrayBuffer(256);
let nodesBuffer = new ArrayBuffer(100_000 * WGSL_NODE_BYTESIZE);
let nodesGpuBuffer = null;
let uniformsGpuBuffer = null;
let layout_0 = null;
let layout_1 = null;
let bindGroup_0 = null;
let stateCache = new Map();

let defaultTexture = null;
let defaultSampler = null;

// some reusable variables to reduce GC strain
let _fm        = new Matrix4();
let _frustum   = new Frustum();
let _world     = new Matrix4();
let _worldView = new Matrix4();
let _pos       = new Vector4();
let _pos2      = new Vector4();
let _box       = new Box3();

function init(renderer){

	if(initialized){
		return;
	}
	
	let {device} = renderer;

	uniformsGpuBuffer = device.createBuffer({
		size: uniformsBuffer.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	nodesGpuBuffer = device.createBuffer({
		size: nodesBuffer.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});

	layout_0 = renderer.device.createBindGroupLayout({
		label: "3d tiles uniforms",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'uniform'},
			},{
				binding: 1,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			}
		],
	});

	layout_1 = renderer.device.createBindGroupLayout({
		label: "3d tiles node data",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			},{
				binding: 1,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				texture: {sampleType: "float", viewDimension: "2d", multisampled: false},
			},{
				binding: 2,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				sampler: {type: 'filtering'},
			}
		],
	});

	bindGroup_0 = device.createBindGroup({
		layout: layout_0,
		entries: [
			{binding: 0, resource: {buffer: uniformsGpuBuffer}},
			{binding: 1, resource: {buffer: nodesGpuBuffer}},
		],
	});

	let module = device.createShaderModule({code: shaderCode});

	pipeline = device.createRenderPipeline({
		layout: device.createPipelineLayout({
			bindGroupLayouts: [
				layout_0, layout_1
			],
		}),
		vertex: {
			module,
			entryPoint: "main_vertex",
			buffers: []
		},
		fragment: {
			module,
			entryPoint: "main_fragment",
			targets: [
				{format: "bgra8unorm"},
				{format: "r32uint"},
			],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	initialized = true;
}

function getState(node, renderer){

	let {device} = renderer; 

	if(!stateCache.has(node)){

		let nodeBuffer = node.content.b3dm.gltf.buffer;
		let gpuBuffer = renderer.getGpuBuffer(nodeBuffer);
		
		// let bindGroup = device.createBindGroup({
		// 	layout: layout_1,
		// 	entries: [
		// 		{binding: 0, resource: {buffer: gpuBuffer}},
		// 	],
		// });

		// let state = {gpuBuffer, bindGroup};
		let state = {gpuBuffer};

		stateCache.set(node, state);
	}

	return stateCache.get(node);
}

export class BVSphere{
	constructor(){
		this.position = new Vector3(0, 0, 0);
		this.radius = 1;
	}
}

let globalNodeCounter = 0;

export class TDTilesNode{

	constructor(){
		this.boundingVolume = new BVSphere();
		this.children = [];
		this.content = null;
		this.contentLoaded = false;
		this.isLoading = false;
		this.tilesetUri = "";
		this.index = globalNodeCounter;
		this.level = 0;
		this.localIndex = 0;
		this.id = "r";

		globalNodeCounter++;
	}

	traverse(callback){

		let keepGoing = callback(this);

		if(!keepGoing) return;

		for(let child of this.children){
			child.traverse(callback);
		}

	}

}

export class TDTiles extends SceneNode{

	constructor(url){
		super(); 

		this.url = url;
		this.dispatcher = new EventDispatcher();
		this.root = new TDTilesNode();
		this.visibleNodes = [];

		this.positions = new Float32Array([
			0.2, 0.2, 0.0,
			0.4, 0.2, 0.0,
			0.4, 0.4, 0.0,
			0.2, 0.2, 0.0,
			0.4, 0.4, 0.0,
			0.2, 0.4, 0.0,
		]);
	}

	setHovered(index){
		// this.hoveredIndex = index;
		// this.dispatcher.dispatch("hover", {
		// 	images: this,
		// 	index: index,
		// 	image: this.images[index],
		// });
	}

	updateUniforms(drawstate){

		let {renderer, camera} = drawstate;
		let {device} = renderer;

		let f32 = new Float32Array(uniformsBuffer);
		let view = new DataView(uniformsBuffer);

		{ // transform
			this.updateWorld();
			let world = this.world;
			let view = camera.view;
			_worldView.multiplyMatrices(view, world);

			f32.set(_worldView.elements, 0);
			f32.set(camera.proj.elements, 16);
		}

		{ // misc
			let size = renderer.getSize();

			view.setFloat32(128, size.width, true);
			view.setFloat32(132, size.height, true);
			view.setFloat32(136, 10.0, true);
			view.setUint32(140, Potree.state.renderedElements, true);
			view.setInt32(144, this.hoveredIndex ?? -1, true);
		}

		renderer.device.queue.writeBuffer(uniformsGpuBuffer, 0, uniformsBuffer, 0, uniformsBuffer.byteLength);
	}

	updateNodesBuffer(drawstate){

		let {renderer, camera} = drawstate;
		let {device} = renderer;

		let bufferView = new DataView(nodesBuffer);
		let f32 = new Float32Array(nodesBuffer);

		_world.makeIdentity();
		let view = camera.view;
		_worldView.makeIdentity();

		let numNodes = this.visibleNodes.length;

		for(let i = 0; i < numNodes; i++){
			let node = this.visibleNodes[i];

			_world.elements[12] = node.boundingVolume.position.x;
			_world.elements[13] = node.boundingVolume.position.y;
			_world.elements[14] = node.boundingVolume.position.z;

			_worldView.multiplyMatrices(view, _world);
			f32.set(_worldView.elements, i * 20);

			if(node?.content?.b3dm){

				let b3dm = node.content.b3dm;

				_world.elements[12] = b3dm.json.RTC_CENTER[0];
				_world.elements[13] = b3dm.json.RTC_CENTER[1];
				_world.elements[14] = b3dm.json.RTC_CENTER[2];

				_worldView.multiplyMatrices(view, _world);
				f32.set(_worldView.elements, i * 20);

				let binStart = b3dm.gltf.chunks[1].start;

				let json = b3dm.gltf.json;
				let indexBufferRef  = json.meshes[0].primitives[0].indices;
				let POSITION_bufferRef = json.meshes[0].primitives[0].attributes.POSITION;
				let TEXCOORD_bufferRef = json.meshes[0].primitives[0].attributes.TEXCOORD_0;

				let index_accessor      = json.accessors[indexBufferRef];
				let POSITION_accessor   = json.accessors[POSITION_bufferRef];
				let TEXCOORD_accessor   = json.accessors[TEXCOORD_bufferRef];

				let index_bufferView    = json.bufferViews[index_accessor.bufferView];
				let POSITION_bufferView = json.bufferViews[POSITION_accessor.bufferView];
				let TEXCOORD_bufferView = json.bufferViews[TEXCOORD_accessor.bufferView];

				// debugger;

				bufferView.setUint32(80 * i + 64 +  0, binStart + 8 + index_bufferView.byteOffset, true);
				bufferView.setUint32(80 * i + 64 +  4, binStart + 8 + POSITION_bufferView.byteOffset, true);
				bufferView.setUint32(80 * i + 64 +  8, binStart + 8 + TEXCOORD_bufferView.byteOffset, true);
				bufferView.setUint32(80 * i + 64 + 12, node.index, true);
			}


		}

		renderer.device.queue.writeBuffer(nodesGpuBuffer, 0, nodesBuffer, 0, 80 * numNodes);
	}

	updateVisibility(renderer, camera){

		let loadQueue = [];

		let view = camera.view;
		let proj = camera.proj;
		_fm.multiplyMatrices(proj, view);
		_frustum.setFromMatrix(_fm);
		
		let screenSize = renderer.getSize();
		
		this.visibleNodes = [];

		// console.log("==== update visibility ==== ");
		this.root.traverse(node => {

			let pixelSize = 0;
			let sse       = 0;
			let bv        = node.boundingVolume;

			if(bv instanceof BVSphere){

				_box.min.x = bv.position.x - 0.5 * bv.radius;
				_box.min.y = bv.position.y - 0.5 * bv.radius;
				_box.min.z = bv.position.z - 0.5 * bv.radius;
				_box.max.x = bv.position.x + 0.5 * bv.radius;
				_box.max.y = bv.position.y + 0.5 * bv.radius;
				_box.max.z = bv.position.z + 0.5 * bv.radius;

				let inFrustum = _frustum.intersectsBox(_box);

				if(!inFrustum) return false;

				_pos.set(bv.position.x, bv.position.y, bv.position.z, 1);
				_pos.applyMatrix4(view);
				_pos2.copy(_pos);
				_pos2.x += node.geometricError;
				_pos.applyMatrix4(proj);
				_pos2.applyMatrix4(proj);

				let dx = (_pos.x / _pos.w) - (_pos2.x / _pos2.w);
				sse = Math.abs(dx * screenSize.width);

				let distance = _pos.w;

				pixelSize = screenSize.width * bv.radius / distance;
			}

			let needsRefinement = sse > 5;

			// DEBUG
			// let strLabel = ("    ".repeat(node.level) + node.id).padStart(10);
			// let strSSE = sse.toFixed(1).padStart(7);
			// let strRefine = needsRefinement ? "yes" : " no";
			// let strContent = node.content == null ? "none" : node.content.uri;
			// console.log(`${strLabel}, sse: ${strSSE}, refine: ${strRefine}, content: ${strContent}`);


			if(needsRefinement){

				let hasChildren = node.children.length > 0;
				let allChildrenLoaded = true;
				for(let child of node.children){
					let childLoaded = child.content == null ? true : child.contentLoaded;

					allChildrenLoaded = allChildrenLoaded && childLoaded;
				}

				if(hasChildren && allChildrenLoaded){
					// keep traversing to show descendants
					return true;
				}else{
					// descendants are not yet ready, so try showing current node
					// and loading descendants
					
					let hasContent = node.content != null;
					let nodeIsLoaded = hasContent && node.contentLoaded;

					if(hasContent && nodeIsLoaded){
						this.visibleNodes.push(node);

						// keep traversing
						return true;
					}else if(hasContent && !nodeIsLoaded){
						loadQueue.push(node);
					}else if(!hasContent){
						// keep traversing
						return true;
					}else{
						// shouldnt happen?
						debugger;
					}
				}
			}else{

				let hasContent = node.content != null;
				let nodeIsLoaded = hasContent && node.contentLoaded;

				if(hasContent && nodeIsLoaded){
					this.visibleNodes.push(node);

					// stop traversing
					return false;
				}else if(hasContent && !nodeIsLoaded){
					loadQueue.push(node);
				}

				// stop traversing
				return false;
			}

			return true;
		});

		for(let i = 0; i < loadQueue.length; i++){

			let node = loadQueue[i];
			this.loader.loadNode(node);

			if(i > 10) break;
		}


	}

	render(drawstate){

		let {renderer, camera} = drawstate;
		let {device} = renderer;

		this.updateVisibility(renderer, camera);

		init(renderer);

		this.updateUniforms(drawstate);
		this.updateNodesBuffer(drawstate);



		if(!defaultTexture){
			let array = new Uint8Array([255, 0, 0, 255]);
			defaultTexture = renderer.createTextureFromArray(array, 1, 1);
		}

		if(!defaultSampler){
			defaultSampler = device.createSampler({
				magFilter    : "linear",
				minFilter    : "linear",
				mipmapFilter : 'linear',
			});
		}


		let {passEncoder} = drawstate.pass;

		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, bindGroup_0);



		// let nodesBindGroup = renderer.device.createBindGroup({
		// 	layout: layout_nodes,
		// 	entries: [
		// 		{binding: 0, resource: {buffer: nodesGpuBuffer}},
		// 	],
		// });

		// passEncoder.setBindGroup(1, nodesBindGroup);

		// let vboPosition = renderer.getGpuBuffer(this.positions);

		// passEncoder.setVertexBuffer(0, vboPosition);

		// let numVertices = this.positions.length / 3;
		// passEncoder.draw(6, 1, 0, 0);

		// Potree.state.renderedElements += numVertices;
		// Potree.state.renderedObjects.push({node: this, numElements: numVertices});

		for(let i = 0; i < this.visibleNodes.length; i++){

			let node = this.visibleNodes[i];

			if(node.content && node.content.b3dm){
				let state = getState(node, renderer);

				let gltf = node.content.b3dm.gltf;
				let json = node.content.b3dm.gltf.json;
				let indexBufferRef  = json.meshes[0].primitives[0].indices;
				// let POSITION_bufferRef = json.meshes[0].primitives[0].attributes.POSITION;
				// let TEXCOORD_bufferRef = json.meshes[0].primitives[0].attributes.TEXCOORD_0;


				if(gltf.image && !node.texture){

					let image = gltf.image;
					let args = {format: "rgba8unorm"};
					let texture = renderer.createTexture(image.width, image.height, args);
					node.texture = texture;

					device.queue.copyExternalImageToTexture(
						{source: gltf.image},
						{texture: texture},
						[image.width, image.height]
					);

					
				}




				let index_accessor      = json.accessors[indexBufferRef];
				// let POSITION_accessor   = json.accessors[TEXCOORD_bufferRef];
				// let TEXCOORD_accessor   = json.accessors[TEXCOORD_bufferRef];

				// let index_bufferView    = json.bufferViews[index_accessor.bufferView];
				// let POSITION_bufferView = json.bufferViews[POSITION_accessor.bufferView];
				// let TEXCOORD_bufferView = json.bufferViews[TEXCOORD_accessor.bufferView];

				let numIndices = index_accessor.count;

				let texture = node.texture ?? defaultTexture;

				let bindGroup1 = device.createBindGroup({
					layout: layout_1,
					entries: [
						{binding: 0, resource: {buffer: state.gpuBuffer}},
						{binding: 1, resource: texture.createView()},
						{binding: 2, resource: defaultSampler},
					],
				});

				passEncoder.setBindGroup(1, bindGroup1);

				passEncoder.draw(numIndices, 1, 0, i);

				// draw bounding box
				// let color = new Vector3(0, 255, 0);
				// let size = node.boundingVolume.radius;
				// renderer.drawBoundingBox(
				// 	node.boundingVolume.position,
				// 	new Vector3(1, 1, 1).multiplyScalar(size),
				// 	color,
				// );
			}else{

			}

			// let num
		}

		// this.root.traverse(node => {
		// 	renderer.drawBoundingBox(
		// 		node.boundingVolume.position,
		// 		new Vector3(1, 1, 1).multiplyScalar(node.boundingVolume.radius),
		// 		node.dbgColor,
		// 	);
		// });

	}


}