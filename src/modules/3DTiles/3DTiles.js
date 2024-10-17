
import {SceneNode, Vector3, Vector4, Matrix4, Box3, Frustum, EventDispatcher, StationaryControls} from "potree";
import {LRUItem, LRU} from "potree";

const WGSL_NODE_BYTESIZE = 80 + 16;
const TILES_CACHE_THRESHOLD = 200_000_000; // If we load more bytes, we start unloading least-recently-used nodes

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

let defaultSampler = null;

// some reusable variables to reduce GC strain
let _fm        = new Matrix4();
let _frustum   = new Frustum();
let _world     = new Matrix4();
let _worldView = new Matrix4();
let _rot       = new Matrix4();
let _trans     = new Matrix4();
let _pos       = new Vector4();
let _pos2      = new Vector4();
let _box       = new Box3();
let _dirx      = new Vector3();
let _diry      = new Vector3();
let _dirz      = new Vector3();

let lru = new LRU();

function isDescendant(nodeID, potentialDescendantID){
	return nodeID === potentialDescendantID || potentialDescendantID.includes(`${nodeID}_`);
}

async function init(renderer){

	if(initialized){
		return;
	}
	
	let {device} = renderer;

	uniformsGpuBuffer = renderer.createBuffer({
		size: uniformsBuffer.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	nodesGpuBuffer = renderer.createBuffer({
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

	let shaderPath = `${import.meta.url}/../3DTiles.wgsl`;
	let response = await fetch(shaderPath);
	let shaderSource = await response.text();

	let module = device.createShaderModule({code: shaderSource});

	let tStart = Date.now();

	let testPipeline = device.createRenderPipelineAsync({
		label: "3DTiles",
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
	testPipeline.then(pipeline => {
		let duration = Date.now() - tStart;
		console.log(`3D Tiles duration: ${duration / 1000} s`);
	});


	pipeline = device.createRenderPipeline({
		label: "3DTiles",
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
	let duration = Date.now() - tStart;

	initialized = true;
}

function getState(node, renderer){

	let {device} = renderer; 

	if(!stateCache.has(node)){

		let nodeBuffer = node.content.b3dm.gltf.buffer;
		let gpuBuffer = renderer.getGpuBuffer(nodeBuffer);
		
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

	toBox(){

		let box = new Box3();

		box.min.x = this.position.x - this.radius;
		box.min.y = this.position.y - this.radius;
		box.min.z = this.position.z - this.radius;

		box.max.x = this.position.x + this.radius;
		box.max.y = this.position.y + this.radius;
		box.max.z = this.position.z + this.radius;

		return box;
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
		this.tdtile = null;
		this.world = new Matrix4();
		this.last_used_in_frame = 0;

		this.projected_dirx = null;
		this.projected_diry = null;
		this.projected_dirz = null;
		this.projected_pos  = null;

		globalNodeCounter++;
	}

	// getBoundingVolumeAsBox(){
	// 	let volume = this.boundingVolume;
	// 	let box = new Box3();

	// 	if(volume instanceof BVSphere){
	// 		let radius = volume.sphere[3];

	// 		box.min.x = volume.sphere[0] - radius;
	// 		box.min.y = volume.sphere[1] - radius;
	// 		box.min.z = volume.sphere[2] - radius;

	// 		box.max.x = volume.sphere[0] + radius;
	// 		box.max.y = volume.sphere[1] + radius;
	// 		box.max.z = volume.sphere[2] + radius;
	// 	}else{
	// 		throw "not implemented";
	// 	}

	// 	return box;
	// }

	traverse(callback){

		let keepGoing = callback(this);

		if(!keepGoing) return;

		for(let child of this.children){
			child.traverse(callback);
		}

	}

}

let bufferBindGroupCache = new Map();

function getCachedBufferBindGroup(renderer, node, layout, gpuBuffer, texture, sampler){

	let cached = bufferBindGroupCache.get(node);

	let isOutdated = cached && (cached.gpuBuffer !== gpuBuffer || cached.texture !== texture);
	if(isOutdated){
		bufferBindGroupCache.delete(node);
		cached = null;
	}

	if(cached){
		return cached.bindGroup;
	}else{

		let bindGroup = renderer.device.createBindGroup({
			layout: layout,
			entries: [
				{binding: 0, resource: {buffer: gpuBuffer}},
				{binding: 1, resource: texture.createView()},
				{binding: 2, resource: sampler},
			],
		});

		let cached = {bindGroup, gpuBuffer, texture};

		bufferBindGroupCache.set(node, cached);

		return bindGroup;
	}

}

export class TDTiles extends SceneNode{

	constructor(url){
		super(); 

		this.url = url;
		this.dispatcher = new EventDispatcher();
		this.root = new TDTilesNode();
		this.root.tdtile = this;
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

		let view = camera.view;
		_world.makeIdentity();
		_worldView.makeIdentity();

		let numNodes = this.visibleNodes.length;

		let meshCounter = 0;     // some nodes have multiple meshes
		let triangleCounter = 0;
		for(let nodeIndex = 0; nodeIndex < numNodes; nodeIndex++){
			let node = this.visibleNodes[nodeIndex];

			if(!node.projected_pos){

				node.projected_dirx = new Vector3();
				node.projected_diry = new Vector3();
				node.projected_dirz = new Vector3();
				node.projected_pos  = new Vector3();

				node.projected_dirx.set(...this.project([
					node.boundingVolume.position.x + 1.0,
					node.boundingVolume.position.y + 0.0,
					node.boundingVolume.position.z + 0.0,
				]), 1);
				node.projected_diry.set(...this.project([
					node.boundingVolume.position.x + 0.0,
					node.boundingVolume.position.y + 1.0,
					node.boundingVolume.position.z + 0.0,
				]), 1);
				node.projected_dirz.set(...this.project([
					node.boundingVolume.position.x + 0.0,
					node.boundingVolume.position.y + 0.0,
					node.boundingVolume.position.z + 1.0,
				]), 1);
				node.projected_pos.set(...this.project([
					node.boundingVolume.position.x,
					node.boundingVolume.position.y,
					node.boundingVolume.position.z,
				]), 1);

				node.projected_dirx.set(
					node.projected_dirx.x - node.projected_pos.x,
					node.projected_dirx.y - node.projected_pos.y,
					node.projected_dirx.z - node.projected_pos.z,
				);
				node.projected_diry.set(
					node.projected_diry.x - node.projected_pos.x,
					node.projected_diry.y - node.projected_pos.y,
					node.projected_diry.z - node.projected_pos.z,
				);
				node.projected_dirz.set(
					node.projected_dirz.x - node.projected_pos.x,
					node.projected_dirz.y - node.projected_pos.y,
					node.projected_dirz.z - node.projected_pos.z,
				);
				node.projected_dirx.normalize();
				node.projected_diry.normalize();
				node.projected_dirz.normalize();
			}

			_rot.makeIdentity();
			_rot.set(
				node.projected_dirx.x, node.projected_diry.x, node.projected_dirz.x, 0.0,
				node.projected_dirx.y, node.projected_diry.y, node.projected_dirz.y, 0.0,
				node.projected_dirx.z, node.projected_diry.z, node.projected_dirz.z, 0.0,
				      0,       0,       0, 1.0,
			);

			_trans.makeIdentity();
			_trans.elements[12] = node.projected_pos.x;
			_trans.elements[13] = node.projected_pos.y;
			_trans.elements[14] = node.projected_pos.z;

			_world.multiplyMatrices(_trans, _rot);
			_worldView.multiplyMatrices(view, _world);

			node.world.elements.set(_world.elements);

			if(node?.content?.b3dm){

				let b3dm = node.content.b3dm;

				if(!b3dm._pos){
					b3dm._pos = new Vector3();

					b3dm._pos.set(...this.project([
						b3dm.json.RTC_CENTER[0],
						b3dm.json.RTC_CENTER[1],
						b3dm.json.RTC_CENTER[2],
					]), 1);
				}

				// _pos.set(...this.project([
				// 	b3dm.json.RTC_CENTER[0],
				// 	b3dm.json.RTC_CENTER[1],
				// 	b3dm.json.RTC_CENTER[2],
				// ]), 1);

				

				_world.elements[12] = b3dm._pos.x;
				_world.elements[13] = b3dm._pos.y;
				_world.elements[14] = b3dm._pos.z;
				_worldView.multiplyMatrices(view, _world);

				let binStart = b3dm.gltf.chunks[1].start;
				let json = b3dm.gltf.json;

				for(let primitive of json.meshes[0].primitives){

					f32.set(_worldView.elements, meshCounter * WGSL_NODE_BYTESIZE / 4);

					let indexBufferRef      = primitive.indices;
					let POSITION_bufferRef  = primitive.attributes.POSITION;
					let TEXCOORD_bufferRef  = primitive.attributes.TEXCOORD_0;

					let index_accessor      = json.accessors[indexBufferRef];
					let POSITION_accessor   = json.accessors[POSITION_bufferRef];
					let TEXCOORD_accessor   = json.accessors[TEXCOORD_bufferRef];

					let index_bufferView    = json.bufferViews[index_accessor.bufferView];
					let POSITION_bufferView = json.bufferViews[POSITION_accessor.bufferView];
					let TEXCOORD_bufferView = json.bufferViews[TEXCOORD_accessor.bufferView];

					bufferView.setUint32(WGSL_NODE_BYTESIZE * meshCounter + 64 +  0, binStart + 8 + index_bufferView.byteOffset, true);
					bufferView.setUint32(WGSL_NODE_BYTESIZE * meshCounter + 64 +  4, binStart + 8 + POSITION_bufferView.byteOffset, true);
					bufferView.setUint32(WGSL_NODE_BYTESIZE * meshCounter + 64 +  8, binStart + 8 + TEXCOORD_bufferView.byteOffset, true);
					bufferView.setUint32(WGSL_NODE_BYTESIZE * meshCounter + 64 + 12, node.index, true);
					bufferView.setUint32(WGSL_NODE_BYTESIZE * meshCounter + 64 + 16, triangleCounter, true);

					let numTriangles = index_accessor.count / 3;
					triangleCounter += numTriangles;
					meshCounter++;
				}
			}


		}

		renderer.device.queue.writeBuffer(nodesGpuBuffer, 0, nodesBuffer, 0, WGSL_NODE_BYTESIZE * meshCounter);
	}

	project(coord){

		if(this.projector){
			return this.projector.forward(coord);
		}else{
			return coord;
		}

	}

	updateVisibility(renderer, camera){

		let loadQueue = [];

		let view = camera.view;
		let proj = camera.proj;
		_fm.multiplyMatrices(proj, view);
		_frustum.setFromMatrix(_fm);
		
		let screenSize = renderer.getSize();
		
		this.visibleNodes = [];

		this.root.traverse(node => {

			// mark this as "recently used"
			node.last_used_in_frame = Potree.state.frameCounter;
			if(node?.content?.b3dm != null && node.contentLoaded){
				lru.touch(node);
			}

			let pixelSize = 0;
			let sse       = 0;
			let bv        = node.boundingVolume;

			if(bv instanceof BVSphere){

				_pos.set(...this.project([
					bv.position.x,
					bv.position.y,
					bv.position.z,
				]), 1);

				_box.min.set(
					_pos.x - 1.0 * bv.radius,
					_pos.y - 1.0 * bv.radius,
					_pos.z - 1.0 * bv.radius,
				);
				_box.max.set(
					_pos.x + 1.0 * bv.radius,
					_pos.y + 1.0 * bv.radius,
					_pos.z + 1.0 * bv.radius,
				);

				let inFrustum = _frustum.intersectsBox(_box);

				if(!inFrustum) return false;

				// _pos.set(bv.position.x, bv.position.y, bv.position.z, 1);
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

			let needsRefinement = sse > 2;
			node.sse = sse;

			// if(node.id === "r170") needsRefinement = false;

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
						// this.visibleNodes.push(node);

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

		// { // DEBUG: Only traverse to specific node
		// 	this.visibleNodes = [];
		// 	let targetNodeID = "r_245_0_0_0_0";
		// 	this.root.traverse(node => {

		// 		let hasContent = node.content != null;
		// 		let nodeIsLoaded = hasContent && node.contentLoaded;

		// 		if(hasContent && nodeIsLoaded){
		// 			this.visibleNodes.push(node);

		// 			return false;
		// 		}else if(hasContent && !nodeIsLoaded){
		// 			loadQueue.push(node);
		// 		}
		// 		
		// 		let keepTraversing = node.id === targetNodeID || targetNodeID.includes(`${node.id}_`);

		// 		return keepTraversing;
		// 	});
		// }

		this.visibleNodes.push(this.root);

		loadQueue.sort((a, b) => {
			return b.sse - a.sse;
		});

		for(let i = 0; i < loadQueue.length; i++){

			let node = loadQueue[i];
			this.loader.loadNode(node);

			if(i > 10) break;
		}

		Potree.debug.lru_tiles = lru;
	}

	disposeLeastRecentlyUsed(renderer){

		let tracker = Potree.resourceTracker["3DTiles"];
		while(tracker.bytesLoaded > TILES_CACHE_THRESHOLD){
			// start unloading least-recently-used nodes

			let item = lru.oldest;
			let node = item.node;

			// console.log(`disposing ${node.id}`);

			// avoid unloading very recently used nodes
			if(node.last_used_in_frame + 5 >= Potree.state.frameCounter){
				break;
			}

			if(node.isLoading) continue;

			let b3dm = node?.content?.b3dm;

			if(b3dm && node.contentLoaded){

				tracker.bytesLoaded -= node.content.b3dm.buffer.byteLength;

				let nodeBuffer = node.content.b3dm.gltf.buffer;
				// let gpuBuffer = renderer.typedBuffers.get(nodeBuffer);
				
				// if(gpuBuffer) gpuBuffer.destroy();
				if(node.textures) node.textures.forEach(texture => {
					renderer.dispose(texture);
				});
				delete node.textures;

				// renderer.typedBuffers.delete(nodeBuffer);
				renderer.disposeGpuBuffer(nodeBuffer);
				stateCache.delete(node);


				delete node.content.b3dm;
				node.contentLoaded = false;
				lru.remove(node);
			}else{
				// Nothing to unload
			}

		}

	}

	render(drawstate){

		let {renderer, camera} = drawstate;
		let {device} = renderer;

		if(Potree.settings.updateEnabled){
			this.updateVisibility(renderer, camera);
		}

		this.disposeLeastRecentlyUsed(renderer);

		// this.visibleNodes = this.visibleNodes.filter(n => n.id === "r_241_0_0_0_0_3_0_0_0_0_0");
		
		if(Potree.settings.dbg3DTile){
			this.visibleNodes = this.visibleNodes.filter(n => isDescendant(n.id, Potree.settings.dbg3DTile));
		}

		init(renderer);

		if(!initialized) return;

		this.updateUniforms(drawstate);
		this.updateNodesBuffer(drawstate);

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

		let meshCounter = 0;
		let triangleCounter = 0;
		for(let nodeIndex = 0; nodeIndex < this.visibleNodes.length; nodeIndex++){

			let node = this.visibleNodes[nodeIndex];

			if(node.content && node.content.b3dm){
				let state = getState(node, renderer);

				let gltf = node.content.b3dm.gltf;
				let json = node.content.b3dm.gltf.json;

				for(let primitiveID = 0; primitiveID < json.meshes[0].primitives.length; primitiveID++)
				{
					let primitive = json.meshes[0].primitives[primitiveID];
					let indexBufferRef  = primitive.indices;

					if(gltf.images && !node.textures){

						node.textures = [];

						for(let image of gltf.images){
							let args = {format: "rgba8unorm", label: `3DTiles texture for node ${node.id}`};
							let texture = renderer.createTexture(image.width, image.height, args);
							node.textures.push(texture);

							device.queue.copyExternalImageToTexture(
								{source: image},
								{texture: texture},
								[image.width, image.height]
							);
						}
					}

					let index_accessor = json.accessors[indexBufferRef];
					let numIndices = index_accessor.count;

					let texture = renderer.defaultTexture;
					if(node.textures){
						texture = node.textures[0];

						if(node.textures.length > primitiveID){
							texture = node.textures[primitiveID];
						}
					}

					// let bindGroup1 = device.createBindGroup({
					// 	layout: layout_1,
					// 	entries: [
					// 		{binding: 0, resource: {buffer: state.gpuBuffer}},
					// 		{binding: 1, resource: texture.createView()},
					// 		{binding: 2, resource: defaultSampler},
					// 	],
					// });
					let bindGroup1 = getCachedBufferBindGroup(renderer, node, layout_1, state.gpuBuffer, texture, defaultSampler);

					passEncoder.setBindGroup(1, bindGroup1);

					passEncoder.draw(numIndices, 1, 0, meshCounter);

					let numTriangles = numIndices / 3;
					Potree.state.renderedElements += numTriangles;
					Potree.state.renderedObjects.push({node: node, numElements: numTriangles});

					triangleCounter += numTriangles;
					meshCounter++;

					Potree.state.num3DTileNodes++;
					Potree.state.num3DTileTriangles += numTriangles;
				}

				// draw bounding box
				if(Potree.settings.showBoundingBox){
					let pos = new Vector3();
					pos.set(...this.project([
						node.boundingVolume.position.x,
						node.boundingVolume.position.y,
						node.boundingVolume.position.z,
					]), 1);
					let color = new Vector3(0, 255, 0);
					let size = node.boundingVolume.radius;
					renderer.drawBoundingBox(
						pos,
						new Vector3(1, 1, 1).multiplyScalar(size),
						color,
					);

					// TODO: would be neat to have helper functions like this:
					// renderer.draw(geometries.boundingSphere, {position: pos, scale: size});
					// renderer.draw(geometries.boundingBox, {position: pos, scale: size});

					renderer.drawSphere(pos, 0.6 * size);
				}

			}else{
				// draw bounding box
				if(Potree.settings.showBoundingBox){
					let pos = new Vector3();
					pos.set(...this.project([
						node.boundingVolume.position.x,
						node.boundingVolume.position.y,
						node.boundingVolume.position.z,
					]), 1);
					let color = new Vector3(255, 0, 0);
					let size = node.boundingVolume.radius;
					// renderer.drawBoundingBox(
					// 	pos,
					// 	new Vector3(1, 1, 1).multiplyScalar(size),
					// 	color,
					// );

					renderer.drawSphere(pos, 0.6 * size);
				}
			}

		}

		if(this.isHighlighted){
			let pos = this.boundingBox.center();
			let size = this.boundingBox.size();
			let color = new Vector3(255, 0, 0);

			renderer.drawBoundingBox(pos, size, color,);
		}
	}


}