
import {PointCloudOctree, REFINEMENT, PointCloudOctreeNode} from "potree";
import {PointCloudMaterial} from "potree";
import {PointAttribute, PointAttributes, PointAttributeTypes} from "potree";
import {WorkerPool} from "potree";
import {Geometry} from "potree";
import {Vector3, Box3, Matrix4} from "potree";
import JSON5 from "json5";
import {MAPPINGS} from "potree";

let numActiveRequests = 0;

const NodeType = {
	NORMAL: 0,
	LEAF: 1,
	PROXY: 2,
};

let typenameTypeattributeMap = {
	"double": PointAttributeTypes.DOUBLE,
	"float": PointAttributeTypes.FLOAT,
	"int8": PointAttributeTypes.INT8,
	"uint8": PointAttributeTypes.UINT8,
	"int16": PointAttributeTypes.INT16,
	"uint16": PointAttributeTypes.UINT16,
	"int32": PointAttributeTypes.INT32,
	"uint32": PointAttributeTypes.UINT32,
	"int64": PointAttributeTypes.INT64,
	"uint64": PointAttributeTypes.UINT64,
};

let tmpVec3 = new Vector3();
function createChildAABB(aabb, index){
	let min = aabb.min.clone();
	let max = aabb.max.clone();
	let size = tmpVec3.copy(max).sub(min);

	if ((index & 0b0001) > 0) {
		min.z += size.z / 2;
	} else {
		max.z -= size.z / 2;
	}

	if ((index & 0b0010) > 0) {
		min.y += size.y / 2;
	} else {
		max.y -= size.y / 2;
	}
	
	if ((index & 0b0100) > 0) {
		min.x += size.x / 2;
	} else {
		max.x -= size.x / 2;
	}

	return new Box3(min, max);
}

function round4(number){
	return number + (4 - (number % 4));
}

let SPECTRAL = [
	[213,62,79],
	[252,141,89],
	[254,224,139],
	[230,245,152],
	[153,213,148],
	[50,136,189],
];

function parseAttributes(jsonAttributes){


	let replacements = {
		"rgb": "rgba",
	};

	let attributeList = [];

	for(let jsonAttribute of jsonAttributes){
		let {name, description, size, numElements, elementSize, min, max, scale, offset} = jsonAttribute;

		let type = typenameTypeattributeMap[jsonAttribute.type];

		let potreeAttributeName = replacements[name] ? replacements[name] : name;

		let attribute = new PointAttribute(potreeAttributeName, type, numElements);

		if(numElements === 1){
			attribute.range = [min[0], max[0]];
		}else{
			attribute.range = [min, max];
		}
		
		attribute.initialRange = attribute.range;
		attribute.description = description;
		attribute.scale = scale;
		attribute.offset = offset;

		attributeList.push(attribute);
	}
	
	let attributes = new PointAttributes(attributeList);

	return attributes;
}

export class Potree3Loader{

	static numBatchesLoading = 0;
	static numLeavesLoading = 0;

	constructor(){
		this.metadata = null;
		this.metanodeMap = new Map();
		this.batchnodeMap = new Map();
		this.nodeMap = new Map();
		this.batches = new Map();
		this.octree = null;
	}

	parseHierarchy(node, buffer){
		
		let view = new DataView(buffer);

		let bytesPerNode = 38;
		let numNodes = buffer.byteLength / bytesPerNode;

		let nodes = new Array(numNodes);
		nodes[0] = node;
		let nodePos = 1;

		for(let i = 0; i < numNodes; i++){
			let current = nodes[i];

			if(!current){
				console.log(nodes);
				debugger;
			}

			let type                  = view.getUint8(i * bytesPerNode + 0);
			let childMask             = view.getUint8(i * bytesPerNode + 1);
			let numElements           = view.getUint32(i * bytesPerNode + 2, true);
			let byteOffset            = Number(view.getBigInt64(i * bytesPerNode + 6, true));
			let byteOffset_unfiltered = Number(view.getBigInt64(i * bytesPerNode + 14, true));
			let byteSize              = Number(view.getUint32(i * bytesPerNode + 22, true));
			let byteSize_position     = Number(view.getUint32(i * bytesPerNode + 26, true));
			let byteSize_filtered     = Number(view.getUint32(i * bytesPerNode + 30, true));
			let byteSize_unfiltered   = Number(view.getUint32(i * bytesPerNode + 34, true));

			if(current.nodeType === NodeType.PROXY){
				// replace proxy with real node
				current.byteOffset = byteOffset;
				current.byteSize = byteSize;
				current.numElements = numElements;
				current.byteOffset_unfiltered = byteOffset_unfiltered;
				current.byteSize_unfiltered = byteSize_unfiltered;
			}else if(type === NodeType.PROXY){
				// load proxy
				current.hierarchyByteOffset = byteOffset;
				current.hierarchyByteSize = byteSize;
				current.numElements = numElements;
			}else{
				// load real node 
				current.byteOffset = byteOffset;
				current.byteSize = byteSize;
				current.numElements = numElements;
				current.byteOffset_unfiltered = byteOffset_unfiltered;
				current.byteSize_unfiltered = byteSize_unfiltered;
			}
			
			current.nodeType = type;

			if(current.nodeType === NodeType.LEAF){
				current.unfilteredLoaded = true;
			}

			if(current.nodeType === NodeType.PROXY){
				continue;
			}

			for(let childIndex = 0; childIndex < 8; childIndex++){
				let childExists = ((1 << childIndex) & childMask) !== 0;

				if(!childExists){
					continue;
				}

				let childName = current.name + childIndex;

				let child = new PointCloudOctreeNode(childName);
				child.boundingBox = createChildAABB(current.boundingBox, childIndex);
				child.name = childName;
				child.spacing = current.spacing / 2;
				child.level = current.level + 1;
				child.octree = this.octree;

				current.children[childIndex] = child;
				child.parent = current;

				// console.log(`parsed nodes[${nodePos}] = ${child.name}`);
				nodes[nodePos] = child;
				nodePos++;
			}
		}

	}

	async loadHierarchy(node){
		// console.log(`load hierarchy(${node.name})`);

		let {hierarchyByteOffset, hierarchyByteSize} = node;
		let hierarchyPath = `${this.url}/../hierarchy.bin`;

		let first = hierarchyByteOffset;
		let last = first + hierarchyByteSize - 1;

		let response = await fetch(hierarchyPath, {
			headers: {
				'content-type': 'multipart/byteranges',
				'Range': `bytes=${first}-${last}`,
			},
		});


		let buffer = await response.arrayBuffer();

		let tStart = performance.now();
		this.parseHierarchy(node, buffer);

		let tEnd = performance.now();
		let durationMS = tEnd - tStart;

		let strKB = Math.ceil(hierarchyByteSize / 1024);
		// console.log(`parsed hierarchy (${strKB} kb) in ${durationMS.toFixed(1)} ms`);

		// console.log(node);
	}

	// loads unfiltered voxel data for inner nodes
	async loadNodeUnfiltered(node){

		if(numActiveRequests > 5) return;
		if(!node.loaded) return; // regular filtered data needs to be loaded first
		if(node.unfilteredLoaded) return; 
		if(node.unfilteredLoading) return;
		if(node.loadAttempts > 5) return;
		if(node.nodeType === NodeType.PROXY) return;

		// CHECK IF NODE+CHILDREN IN CONTIGUOUS MEMORY
		// let inContiguousMemory = false;
		// let totalBytes = 0;
		// if((node.level % 2) === 0){
			
		// 	let firstByte = Infinity;
		// 	let lastByte = 0;

		// 	for(let famNode of [node, ...node.children]){

		// 		if(famNode == null) continue;
		// 		if(famNode.nodeType === NodeType.PROXY) continue;
		// 		if(famNode.nodeType === NodeType.LEAF) continue;
		// 		if(famNode.loaded === false) continue;

		// 		firstByte = Math.min(firstByte, famNode.byteOffset_unfiltered);
		// 		lastByte = Math.max(lastByte, famNode.byteOffset_unfiltered + famNode.byteSize_unfiltered);
		// 		totalBytes += famNode.byteSize_unfiltered;
		// 	}

		// 	let diff = lastByte - firstByte;

		// 	inContiguousMemory = (diff === totalBytes);
		// }

		let nodes = [node];

		// if all nodes in chunk of contiguous memory have less than <x> bytes, load them at once
		// if(inContiguousMemory && totalBytes < 1_000_000){
		// 	for(let child of node.children){
		// 		if(child == null) continue;

		// 		child.loading = true;
		// 		nodes.push(child);
		// 	}
		// }

		let chunkOffset = Infinity;
		let chunkSize = 0;

		for(let node of nodes){
			node.unfilteredLoading = true;
			chunkOffset = Math.min(chunkOffset, node.byteOffset_unfiltered);
			chunkSize += node.byteSize_unfiltered;
		}

		if(chunkSize === 0 || chunkSize > 20_000_000){
			debugger;
		}

		let url = new URL(`${this.url}/../octree.bin`, document.baseURI).href;
		numActiveRequests++;
		let promise = fetch(url, {
			headers: {
				'content-type': 'multipart/byteranges',
				'Range': `bytes=${chunkOffset}-${chunkOffset + chunkSize - 1}`,
			},
		});

		let failed = false;
		promise.catch(e => {
			console.log("failed...", e);
			failed = true;
		});

		numActiveRequests--;
		if(failed) return;

		let response = await promise;

		let buffer = await response.arrayBuffer();

		for(let node of nodes){
			let source_u8 = new Uint8Array(buffer, node.byteOffset_unfiltered - chunkOffset, node.byteSize_unfiltered);
			let target_u8 = new Uint8Array(node.geometry.buffer);

			let n = node.numElements;

			let sourceAttOffset = 0;
			let targetAttOffset = 0;
			for(let attribute of this.attributes.attributes){

				let isFiltered = ["position", "rgba"].includes(attribute.name);

				if(isFiltered){
					targetAttOffset += attribute.byteSize;
				}else{
					for(let pointIndex = 0; pointIndex < n; pointIndex++)
					for(let j = 0; j < attribute.byteSize; j++)
					{
						let value = source_u8[n * sourceAttOffset + pointIndex * attribute.byteSize + j];
						target_u8[n * targetAttOffset + pointIndex * attribute.byteSize + j] = value;
					}

					targetAttOffset += attribute.byteSize;
					sourceAttOffset += attribute.byteSize;
				}

			}

			node.unfilteredLoading = false;
			node.unfilteredLoaded = true;
			node.dirty = true;
		}

	}

	// loads filtered voxel data for inner nodes or full point data for leaf nodes
	async loadNode(node){

		const workerPath = "./src/potree/octree/loader_v3/DecoderWorker.js";

		if(WorkerPool.getWorkerCount(workerPath) > 6)
		if(WorkerPool.getAvailableWorkerCount(workerPath) === 0)
		{
			return;
		}
	

		if(node.loaded) return; 
		if(node.loading) return;
		if(node.loadAttempts > 5) return;
		// if(nodesLoading >= 6) return;
		// if(nodesLoading >= MAX_NODES_LOADING) return;
		if(node.parent != null && !node.parent.loaded) return;

		node.loading = true;

		try{
			if(node.nodeType === NodeType.PROXY){
				await this.loadHierarchy(node);
			}
		}catch(e){
			console.log(e);
		}

		// CHECK IF NODE+CHILDREN IN CONTIGUOUS MEMORY
		let inContiguousMemory = false;
		let totalBytes = 0;
		if((node.level % 2) === 0){
			
			let firstByte = Infinity;
			let lastByte = 0;

			for(let famNode of [node, ...node.children]){

				if(famNode == null) continue;
				if(famNode.nodeType === NodeType.PROXY) continue;

				firstByte = Math.min(firstByte, famNode.byteOffset);
				lastByte = Math.max(lastByte, famNode.byteOffset + famNode.byteSize);
				totalBytes += famNode.byteSize;
			}

			let diff = lastByte - firstByte;

			inContiguousMemory = (diff === totalBytes);
		}
		inContiguousMemory = false;

		let nodes = [node];

		// if all nodes in chunk of contiguous memory have less than <x> bytes, load them at once
		if(inContiguousMemory && totalBytes < 1_000_000){
			for(let child of node.children){
				if(child == null) continue;

				child.loading = true;
				nodes.push(child);
			}
		}

		for(let node of nodes){
			node.loading = true;
		}

		try{
			if(node.nodeType === NodeType.PROXY){
				await this.loadHierarchy(node);
			}

			// TODO fix path. This isn't flexible. should be relative from PotreeLoader.js
			
			let worker = WorkerPool.getWorker(workerPath, {type: "module"});

			worker.onmessage = (e) => {
				let data = e.data;

				if(data === "failed"){
					console.log(`failed to load ${node.name}. trying again!`);

					for(let node of nodes){
						node.loaded = false;
						node.loading = false;
					}

					WorkerPool.returnWorker(workerPath, worker);

					return;
				}else if(data?.type === "node parsed"){

					let loadedNode = nodes.find(node => node.name === e.data.name);

					let geometry = new Geometry();
					geometry.numElements = loadedNode.numElements;
					geometry.buffer = data.buffer;
					geometry.statsList = data.statsList;

					// console.log(`loaded ${loadedNode.name}`);

					loadedNode.loaded = true;
					loadedNode.loading = false;
					loadedNode.geometry = geometry;
					loadedNode.voxelCoords = data.voxelCoords;

					if(loadedNode.name === "r"){
						this.octree.events.dispatcher.dispatch("root_node_loaded", {octree: this.octree, loadedNode});
					}
				}else if(e.data === "finished"){
					WorkerPool.returnWorker(workerPath, worker);
				}

			};

			let url = new URL(`${this.url}/../octree.bin`, document.baseURI).href;
			let parentVoxelCoords = node.parent?.voxelCoords;

			let msg_nodes = [];
			for(let node of nodes){

				if(node.numElements === 0){
					debugger;
				}

				let msg_node = {
					name:                  node.name,
					numVoxels:             node.nodeType === NodeType.NORMAL ? node.numElements : 0,
					numPoints:             node.nodeType === NodeType.LEAF ? node.numElements : 0,
					min:                   node.boundingBox.min.toArray(),
					max:                   node.boundingBox.max.toArray(),
					byteOffset:            node.byteOffset,
					byteSize:              node.byteSize,
					byteOffset_unfiltered: node.byteOffset_unfiltered,
					byteSize_unfiltered:   node.byteSize_unfiltered,
				};

				msg_nodes.push(msg_node);
			}

			let msg_octree = {
				min:    this.octree.loader.metadata.boundingBox.min,
				scale:  this.scale,
				offset: this.offset,
				pointAttributes: this.attributes,
			};

			let message = {
				octree: msg_octree,
				nodes: msg_nodes, 
				url, parentVoxelCoords
			};

			let transferables = parentVoxelCoords ? [parentVoxelCoords] : [];

			worker.postMessage(message, transferables.buffer);
			
		}catch(e){
			debugger;
			node.loaded = false;
			node.loading = false;

			console.log(`failed to load ${node.name}`);
			console.log(e);
			console.log(`trying again!`);

			// loading with range requests frequently fails in chrome 
			// loading again usually resolves this.
		}
	}

	static async load(url){
		let loader = new Potree3Loader();
		loader.url = url;

		let response = await fetch(url);
		let text = await response.text();
		let metadata = JSON5.parse(text);

		let attributes = parseAttributes(metadata.attributes);
		loader.metadata = metadata;
		loader.attributes = attributes;
		loader.scale = metadata.scale;
		loader.offset = metadata.offset;

		let octree = new PointCloudOctree();
		octree.url = url;
		octree.spacing = metadata.spacing;
		octree.boundingBox = new Box3(
			new Vector3(...metadata.boundingBox.min),
			new Vector3(...metadata.boundingBox.max),
		);
		octree.position.copy(octree.boundingBox.min);
		octree.boundingBox.max.sub(octree.boundingBox.min);
		octree.boundingBox.min.set(0, 0, 0);
		octree.updateWorld();
		octree.refinement = REFINEMENT.REPLACING;

		octree.attributes = attributes;
		octree.loader = loader;
		loader.octree = octree;
		octree.material.init(octree);

		// add standard attribute mappings
		for(let mapping of Object.values(MAPPINGS)){
			if(["vec3", "scalar"].includes(mapping.name)){
				octree.material.registerMapping(mapping);
			}
		}

		let root = new PointCloudOctreeNode("r");
		root.boundingBox = octree.boundingBox.clone();
		root.level = 0;
		root.nodeType = NodeType.PROXY;
		root.hierarchyByteOffset = 0;
		root.hierarchyByteSize = metadata.hierarchy.firstChunkSize;
		root.spacing = octree.spacing;
		root.byteOffset = 0;
		root.octree = octree;

		loader.loadNode(root);

		octree.root = root;

		Potree.events.dispatcher.dispatch("pointcloud_loaded", octree);

		return octree;
	}

}