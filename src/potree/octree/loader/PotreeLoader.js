
import {PointAttributeTypes, PointAttribute, PointAttributes} from "potree";
import {Vector3, Box3, Matrix4} from "potree";
import {PointCloudOctree, REFINEMENT, PointCloudOctreeNode} from "potree";
import {WorkerPool} from "potree";
import {Geometry} from "potree";
import {MAPPINGS} from "potree";

let nodesLoading = 0;

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

	let hasNX = attributeList.find(a => a.name === "NormalX") != null;
	let hasNY = attributeList.find(a => a.name === "NormalY") != null;
	let hasNZ = attributeList.find(a => a.name === "NormalZ") != null;
	if(hasNX && hasNY && hasNZ){

		let aNormalX = attributeList.find(a => a.name === "NormalX");
		let aNormalY = attributeList.find(a => a.name === "NormalY");
		let aNormalZ = attributeList.find(a => a.name === "NormalZ");
		let aNormal = new PointAttribute("Normal", aNormalX.type, 3);
		aNormal.range = [
			[aNormalX.range[0], aNormalY.range[0], aNormalZ.range[0]],
			[aNormalX.range[1], aNormalY.range[1], aNormalZ.range[1]],
		];

		let indexX = attributeList.indexOf(aNormalX);
		attributeList[indexX] = aNormalX;
		attributeList = attributeList.filter(a => !["NormalX", "NormalY", "NormalZ"].includes(a.name));

		attributeList.push(aNormal);

	}
	
	let attributes = new PointAttributes(attributeList);

	return attributes;
}

export class PotreeLoader{

	constructor(){
		this.metadata = null;
	}

	parseHierarchy(node, buffer){
		
		let view = new DataView(buffer);

		let bytesPerNode = 22;
		let numNodes = buffer.byteLength / bytesPerNode;

		let nodes = new Array(numNodes);
		nodes[0] = node;
		let nodePos = 1;

		for(let i = 0; i < numNodes; i++){
			let current = nodes[i];

			let type = view.getUint8(i * bytesPerNode + 0);
			let childMask = view.getUint8(i * bytesPerNode + 1);
			let numPoints = view.getUint32(i * bytesPerNode + 2, true);
			let byteOffset = Number(view.getBigInt64(i * bytesPerNode + 6, true));
			let byteSize = Number(view.getBigInt64(i * bytesPerNode + 14, true));

			if(current.nodeType === NodeType.PROXY){
				// replace proxy with real node
				current.byteOffset = byteOffset;
				current.byteSize = byteSize;
				current.numPoints = numPoints;
			}else if(type === NodeType.PROXY){
				// load proxy
				current.hierarchyByteOffset = byteOffset;
				current.hierarchyByteSize = byteSize;
				current.numPoints = numPoints;
			}else{
				// load real node 
				current.byteOffset = byteOffset;
				current.byteSize = byteSize;
				current.numPoints = numPoints;
			}
			
			current.nodeType = type;

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

				nodes[nodePos] = child;
				nodePos++;
			}
		}

	}

	async loadHierarchy(node){

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

		this.parseHierarchy(node, buffer);

		// console.log(node);
	}

	async loadNodeUnfiltered(node){
		// this point cloud format does not have dedicated unfiltered buffers
	}

	async loadNode(node){
		
		if(node.loaded) return; 
		if(node.loading) return;
		if(node.loadAttempts > 5) return;
		if(nodesLoading >= 10) return;

		nodesLoading++;
		node.loading = true;

		try{
			if(node.nodeType === NodeType.PROXY){
				await this.loadHierarchy(node);
			}

			// TODO fix path. This isn't flexible. should be relative from PotreeLoader.js
			let workerPath = null;
			if(!this.metadata.encoding || this.metadata.encoding === "DEFAULT"){
				workerPath = "./src/potree/octree/loader/DecoderWorker_default.js";
			}else if(this.metadata.encoding === "BROTLI"){
				workerPath = "./src/potree/octree/loader/DecoderWorker_brotli.js";
			}
			
			let worker = WorkerPool.getWorker(workerPath, {type: "module"});

			worker.onmessage = (e) => {
				let data = e.data;

				if(data === "failed"){
					console.log(`failed to load ${node.name}. trying again!`);

					node.loaded = false;
					node.loading = false;
					nodesLoading--;

					WorkerPool.returnWorker(workerPath, worker);

					return;
				}

				let geometry = new Geometry();
				geometry.numElements = node.numPoints;
				geometry.buffer = data.buffer;
				geometry.statsList = data.statsList;

				node.numElements = node.numPoints;
				node.loaded = true;
				node.loading = false;
				nodesLoading--;
				node.geometry = geometry;

				WorkerPool.returnWorker(workerPath, worker);

				if(node.name === "r"){
					this.octree.events.dispatcher.dispatch("root_node_loaded", {octree: this.octree, node});
				}
			};

			let {byteOffset, byteSize} = node;
			let url = new URL(`${this.url}/../octree.bin`, document.baseURI).href;
			let pointAttributes = this.attributes;

			let {scale, offset} = this;
			let {name, numPoints} = node;
			let min = this.octree.loader.metadata.boundingBox.min;
			let nodeMin = node.boundingBox.min.toArray(); 
			let nodeMax = node.boundingBox.max.toArray();

			let message = {
				name, url, byteOffset, byteSize, numPoints,
				pointAttributes, scale, offset, min, nodeMin, nodeMax
			};

			worker.postMessage(message, []);
			
		}catch(e){
			debugger;
			node.loaded = false;
			node.loading = false;
			nodesLoading--;

			console.log(`failed to load ${node.name}`);
			console.log(e);
			console.log(`trying again!`);

			// loading with range requests frequently fails in chrome 
			// loading again usually resolves this.
		}

	}

	static async load(url){
		let loader = new PotreeLoader();
		loader.url = url;

		let response = await fetch(url);
		let metadata = await response.json();

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
		octree.refinement = REFINEMENT.ADDITIVE;

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

		//loader.loadHierarchy(root);
		loader.loadNode(root);

		octree.root = root;

		Potree.events.dispatcher.dispatch("pointcloud_loaded", octree);

		return octree;
	}

}