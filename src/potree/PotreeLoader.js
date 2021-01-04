
import {PointCloudOctree} from "./PointCloudOctree.js";
import {PointCloudOctreeNode} from "./PointCloudOctreeNode.js";

const NodeType = {
	NORMAL: 0,
	LEAF: 1,
	PROXY: 2,
};

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

		console.log(node);
	}

	async loadNode(node){
		
		if(node.loaded || node.loading){
			return;
		}

		console.log(`load: ${node.name}`);
		node.loading = true;

		try{
			if(node.nodeType === NodeType.PROXY){
				await this.loadHierarchy(node);
			}

			let {byteOffset, byteSize} = node;
			let urlOctree = `${this.url}/../octree.bin`;

			let first = byteOffset;
			let last = byteOffset + byteSize - 1n;

			let buffer;
			if(byteSize === 0n){
				buffer = new ArrayBuffer(0);
				console.warn(`loaded node with 0 bytes: ${node.name}`);
			}else{
				let response = await fetch(urlOctree, {
					headers: {
						'content-type': 'multipart/byteranges',
						'Range': `bytes=${first}-${last}`,
					},
				});

				buffer = await response.arrayBuffer();
			}

			// TODO fix path. This isn't flexible. 
			//new Worker("./src/potree/DecoderWorker_brotli.js",);
			let worker = new Worker("./src/potree/DecoderWorker_brotli.js", { type: "module" });

			worker.onmessage = function(e){
				console.log(e);
			};

			// let pointAttributes = this.pointAttributes;
			// let scale = this.scale;
			// let min = new Vector3(
			// 	this.offset.x + this.boundingBox.x,
			// 	this.offset.y + this.boundingBox.y,
			// 	this.offset.z + this.boundingBox.z,
			// );
			// let size = 

			let message = {
				name: node.name,
				buffer: buffer,
				pointAttributes: pointAttributes,
				scale: scale,
				min: min,
				// max: max,
				// size: size,
				// offset: offset,
				// numPoints: numPoints
			};

			worker.postMessage(message, [message.buffer]);
			
		}catch(e){
			node.loaded = false;
			node.loading = false;

			console.log(`failed to load ${node.name}`);
			console.log(e);
			console.log(`trying again!`);

			// loading with range requests frequently fails in chrome 
			// loading again usually resolves this.
			this.loadNode(node);
		}

	}

	static async load(url){
		let loader = new PotreeLoader();
		loader.url = url;

		let response = await fetch(url);
		let metadata = await response.json();

		let attributes = OctreeLoader.parseAttributes(metadata.attributes);
		loader.metadata = metadata;
		loader.attributes = attributes;
		loader.scale = metadata.scale;
		loader.offset = metadata.offset;

		let octree = new PointCloudOctree();
		octree.url = url;
		octree.spacing = metadata.spacing;

		octree.loader = loader;
		loader.octree = octree;

		let root = new PointCloudOctreeNode("r");
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