
import {PointCloudOctree} from "./PointCloudOctree.js";
import {PointCloudOctreeNode} from "./PointCloudOctreeNode.js";
import {PointAttribute, PointAttributes, PointAttributeTypes} from "./PointAttributes.js";
import {WorkerPool} from "../misc/WorkerPool.js";
import {Geometry} from "../core/Geometry.js";
import {Vector3, Box3, Matrix4} from "../math/math.js";

let nodesLoading = 0;

const NodeType = {
	NORMAL: 0,
	LEAF: 1,
	PROXY: 2,
};

let typenameTypeattributeMap = {
	"double": PointAttributeTypes.DATA_TYPE_DOUBLE,
	"float": PointAttributeTypes.DATA_TYPE_FLOAT,
	"int8": PointAttributeTypes.DATA_TYPE_INT8,
	"uint8": PointAttributeTypes.DATA_TYPE_UINT8,
	"int16": PointAttributeTypes.DATA_TYPE_INT16,
	"uint16": PointAttributeTypes.DATA_TYPE_UINT16,
	"int32": PointAttributeTypes.DATA_TYPE_INT32,
	"uint32": PointAttributeTypes.DATA_TYPE_UINT32,
	"int64": PointAttributeTypes.DATA_TYPE_INT64,
	"uint64": PointAttributeTypes.DATA_TYPE_UINT64,
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

	let attributes = new PointAttributes();

	let replacements = {
		"rgb": "rgba",
	};

	for(let jsonAttribute of jsonAttributes){
		let {name, description, size, numElements, elementSize, min, max} = jsonAttribute;

		let type = typenameTypeattributeMap[jsonAttribute.type];

		let potreeAttributeName = replacements[name] ? replacements[name] : name;

		let attribute = new PointAttribute(potreeAttributeName, type, numElements);

		if(numElements === 1){
			attribute.range = [min[0], max[0]];
		}else{
			attribute.range = [min, max];
		}
		
		attribute.initialRange = attribute.range;

		attributes.add(attribute);
	}

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
			let byteOffset = view.getBigInt64(i * bytesPerNode + 6, true);
			let byteSize = view.getBigInt64(i * bytesPerNode + 14, true);

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
		let last = first + hierarchyByteSize - 1n;

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

	async loadNode(node){
		
		if(node.loaded || node.loading){
			return;
		}

		if(node.loadAttempts > 5){
			// give up if node failed to load multiple times in a row.
			return;
		}

		if(nodesLoading >= 10){
			return;
		}

		nodesLoading++;
		node.loading = true;

		try{
			if(node.nodeType === NodeType.PROXY){
				await this.loadHierarchy(node);
			}

			// TODO fix path. This isn't flexible. should be relative from PotreeLoader.js
			//new Worker("./src/potree/DecoderWorker_brotli.js",);
			//let worker = new Worker("./src/potree/DecoderWorker_brotli.js", { type: "module" });
			let workerPath = "./src/potree/DecoderWorker_brotli.js";
			let worker = WorkerPool.getWorker(workerPath, {type: "module"});

			worker.onmessage = function(e){
				let data = e.data;

				if(data === "failed"){
					console.log(`failed to load ${node.name}. trying again!`);

					node.loaded = false;
					node.loading = false;
					nodesLoading--;

					WorkerPool.returnWorker(workerPath, worker);

					return;
				}

				let attributeBuffers = data.attributeBuffers;

				let buffers = [];

				for(let attributeName in attributeBuffers){
					buffers.push(attributeBuffers[attributeName]);
				}

				let geometry = new Geometry();
				geometry.numElements = node.numPoints;
				geometry.buffers = buffers;

				node.loaded = true;
				node.loading = false;
				nodesLoading--;
				node.geometry = geometry;

				WorkerPool.returnWorker(workerPath, worker);
			};

			let {byteOffset, byteSize} = node;
			let url = new URL(`${this.url}/../octree.bin`, document.baseURI).href;
			let pointAttributes = this.attributes;
			let scale = this.scale;
			let offset = this.offset;
			// let min = [0, 0, 0];
			let min = this.octree.loader.metadata.boundingBox.min;
			let numPoints = node.numPoints;
			let name = node.name;

			let message = {
				name, url, byteOffset, byteSize, numPoints,
				pointAttributes, scale, offset, min,
			};

			worker.postMessage(message, []);
			
		}catch(e){
			node.loaded = false;
			node.loading = false;
			nodesLoading--;

			console.log(`failed to load ${node.name}`);
			console.log(e);
			console.log(`trying again!`);

			// loading with range requests frequently fails in chrome 
			// loading again usually resolves this.
			// this.loadNode(node);
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
		// octree.world = new Matrix4();
		// octree.world.translate(...metadata.boundingBox.min);

		octree.loader = loader;
		loader.octree = octree;

		let root = new PointCloudOctreeNode("r");
		root.boundingBox = octree.boundingBox.clone();
		root.level = 0;
		root.nodeType = NodeType.PROXY;
		root.hierarchyByteOffset = 0n;
		root.hierarchyByteSize = BigInt(metadata.hierarchy.firstChunkSize);
		root.spacing = octree.spacing;
		root.byteOffset = 0;

		//loader.loadHierarchy(root);
		loader.loadNode(root);

		octree.root = root;

		return octree;
	}

}