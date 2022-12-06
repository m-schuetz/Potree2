
import {PointCloudOctree, PointCloudOctreeNode} from "potree";
import {PointAttribute, PointAttributes, PointAttributeTypes} from "./PointAttributes.js";
import {WorkerPool} from "../misc/WorkerPool.js";
import {Geometry} from "potree";
import {Vector3, Box3, Matrix4} from "potree";


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

async function loadHeader(url){

	let response = await fetch(url, {
		headers: {
			'content-type': 'multipart/byteranges',
			'Range': `bytes=0-549`,
		},
	});

	let buffer = await response.arrayBuffer();

	let view = new DataView(buffer);

	let versionMajor = view.getUint8(24);
	let versionMinor = view.getUint8(25);
	let headerSize = view.getUint16(94, true);
	let offsetToPointData = view.getUint32(96, true);
	let numVLRs = view.getUint32(100, true);
	let pointFormat = view.getUint8(104);
	let pointRecordLength = view.getUint16(105, true);
	let numPoints = 0;
	
	if(versionMajor === 1 && versionMinor < 4){
		numPoints = view.getUint32(107, true);
	}else{
		numPoints = Number(view.getBigUint64(247, true));
	}

	let scale = [
		view.getFloat64(131, true),
		view.getFloat64(139, true),
		view.getFloat64(147, true),
	];

	let offset = [
		view.getFloat64(155, true),
		view.getFloat64(163, true),
		view.getFloat64(171, true),
	];

	let min = [
		view.getFloat64(187, true),
		view.getFloat64(203, true),
		view.getFloat64(219, true),
	];

	let max = [
		view.getFloat64(179, true),
		view.getFloat64(195, true),
		view.getFloat64(211, true),
	];

	let startFirstEVLR = Number(view.getBigUint64(235, true));
	let numEVLRs = view.getUint32(243, true);


	let header = {
		versionMajor,
		versionMinor,
		headerSize,
		offsetToPointData,
		numVLRs,
		pointFormat,
		pointRecordLength,
		numPoints,
		scale, 
		offset, 
		min, max,
		startFirstEVLR, numEVLRs
	};

	return header;
}

function readString(buffer, offset, length){

	let view = new Uint8Array(buffer);
	let string = "";

	for(let i = 0; i < length; i++){
		let char = view[offset + i];

		if(char === 0){
			break;
		}

		string = string + String.fromCharCode(char);
	}

	return string;
}

async function loadVLRs(url, header){

	let response = await fetch(url, {
		headers: {
			'content-type': 'multipart/byteranges',
			'Range': `bytes=${header.headerSize}-${header.offsetToPointData - 1}`,
		},
	});

	let buffer = await response.arrayBuffer();
	let view = new DataView(buffer);

	let vlrs = [];
	let offset = 0;

	for(let i = 0; i < header.numVLRs; i++){

		if(offset >= buffer.byteLength){
			console.warn(`Stopped reading at VLR[${i}] because offset(${offset}) is >= vlrBufferSize(${buffer.byteLength})`);
			break;
		}

		let reserved = view.getUint16(offset + 0, true);
		let userID = readString(buffer, 2, 16);
		let recordID = view.getUint16(offset + 18, true);
		let recordLength = view.getUint16(offset + 20, true);
		let description = readString(buffer, offset + 22, 32);

		let VLR = {
			userID, recordID, recordLength, description,
			buffer: buffer.slice(offset + 54, offset + 54 + recordLength),
		};

		offset = offset + 54 + recordLength;

		vlrs.push(VLR);
	}

	return vlrs;
}

function parseCopcInfo(vlrs){

	let vlr = vlrs.find(vlr => vlr.recordID === 1 && vlr.userID == "copc");

	if(!vlr){
		return null;
	}

	let view = new DataView(vlr.buffer);

	let center = new Vector3(
		view.getFloat64(0, true),
		view.getFloat64(8, true),
		view.getFloat64(16, true),
	);

	let halfsize = view.getFloat64(24, true);
	let spacing = view.getFloat64(32, true);
	let root_hier_offset = Number(view.getBigUint64(40, true));
	let root_hier_size = Number(view.getBigUint64(48, true));
	let gpstime_minimum = view.getFloat64(56, true);
	let gpstime_maximum = view.getFloat64(64, true);

	let copcInfo = {
		center, halfsize, spacing, 
		root_hier_offset, root_hier_size, 
		gpstime_minimum, gpstime_maximum, 
	};

	return copcInfo;
}

function eptKeyToPotreeKey(level, x, y, z){

	let potreeKey = "r";

	for(let i = 1; i <= level; i++){
		let shift = level - i;

		let ix = (x >> shift) & 1;
		let iy = (y >> shift) & 1;
		let iz = (z >> shift) & 1;

		let childIndex = (ix << 2) | (iy << 1) | iz;

		potreeKey = potreeKey + childIndex;
	}

	return potreeKey;
}

function parseAttributes(header, vlrs){

	// TODO: read extra attributes from vlr

	let format = header.pointFormat % 128;
	let types = PointAttributeTypes;

	let baseAttributesMap = {
		6: [
			new PointAttribute("XYZ"                   , types.INT32  , 3),
			new PointAttribute("intensity"             , types.UINT16 , 1),
			new PointAttribute("return number"         , types.UINT8  , 1),
			new PointAttribute("classification flags"  , types.UINT8  , 1),
			new PointAttribute("classification"        , types.UINT8  , 1),
			new PointAttribute("user data"             , types.UINT8  , 1),
			new PointAttribute("scan angle"            , types.INT16  , 1),
			new PointAttribute("point source id"       , types.UINT16 , 1),
			new PointAttribute("gps-time"              , types.DOUBLE , 1),
		],
		7: [
			new PointAttribute("XYZ"                   , types.INT32  , 3),
			new PointAttribute("intensity"             , types.UINT16 , 1),
			new PointAttribute("return number"         , types.UINT8  , 1),
			new PointAttribute("classification flags"  , types.UINT8  , 1),
			new PointAttribute("classification"        , types.UINT8  , 1),
			new PointAttribute("user data"             , types.UINT8  , 1),
			new PointAttribute("scan angle"            , types.INT16  , 1),
			new PointAttribute("point source id"       , types.UINT16 , 1),
			new PointAttribute("gps-time"              , types.DOUBLE , 1),
			new PointAttribute("rgba"                  , types.UINT16 , 3),
		],
	};

	let pointAttributes = new PointAttributes();
	let attributes = baseAttributesMap[format];

	if(!attributes){
		throw "unable to parse point attributes";
	}

	for(let attribute of attributes){
		pointAttributes.add(attribute);
	}

	return attributes;
}



export class CopcLoader{

	constructor(){
		this.header = null;
	}

	parseHierarchy(localRoot, buffer){

		let numEntries = buffer.byteLength / 32;
		let view = new DataView(buffer);

		let nodeMap = new Map();
		nodeMap.set(localRoot.name, localRoot);

		// 1.PASS: read node data
		for(let i = 0; i < numEntries; i++){
			
			let level = view.getUint32(32 * i + 0, true);
			let x = view.getUint32(32 * i + 4, true);
			let y = view.getUint32(32 * i + 8, true);
			let z = view.getUint32(32 * i + 12, true);

			let offset = Number(view.getBigUint64(32 * i + 16, true));
			let byteSize = view.getUint32(32 * i + 24, true);
			let pointCount = view.getInt32(32 * i + 28, true);

			let nodeName = eptKeyToPotreeKey(level, x, y, z);

			let node = nodeMap.get(nodeName);
			if(!node){
				node = new PointCloudOctreeNode(nodeName);
				node.octree = this.octree;
				nodeMap.set(node.name, node);
			}

			node.level = node.name.length - 1;
			node.numPoints = pointCount;
			node.byteOffset = offset;
			node.byteSize = byteSize;
		}

		// 2.Pass connect nodes 
		for(let [nodeName, node] of nodeMap){

			let parentName = node.name.slice(0, -1);
			let parent = nodeMap.get(parentName);
			let childIndex = parseInt(node.name.slice(-1));

			if(parent){
				node.parent = parent;
				parent.children[childIndex] = node;
			}
		}

		// 3.Pass compute derivatives (bounding boxes, ...)
		localRoot.traverse(node => {

			if(node.parent){
				let childIndex = parseInt(node.name.slice(-1));
				let boundingBox = createChildAABB(node.parent.boundingBox, childIndex);

				node.boundingBox = boundingBox;
				node.spacing = node.parent.spacing / 2;

				node.parent.nodeType = NodeType.NORMAL;
			}

			// mark as PROXY if pointCount is -1 (more hierarchy to load)
			// otherwise it's a LEAF node
			if(node.numPoints === -1){
				node.nodeType = NodeType.PROXY;
				node.hierarchyByteOffset = node.byteOffset;
				node.hierarchyByteSize = node.byteSize;
			}else{
				node.nodeType = NodeType.LEAF;
			}
			

		});

	}

	async loadHierarchy(node){
		let {hierarchyByteOffset, hierarchyByteSize} = node;

		let first = hierarchyByteOffset;
		let last = first + hierarchyByteSize - 1;

		let response = await fetch(this.url, {
			headers: {
				'content-type': 'multipart/byteranges',
				'Range': `bytes=${first}-${last}`,
			},
		});

		let buffer = await response.arrayBuffer();

		this.parseHierarchy(node, buffer);
	}

	async loadNode(node){
		if(node.loaded || node.loading){
			return;
		}

		if(node.loadAttempts > 5){
			// give up if node failed to load multiple times in a row.
			return;
		}

		if(nodesLoading >= 6){
			return;
		}

		nodesLoading++;
		node.loading = true;

		try{
			if(node.nodeType === NodeType.PROXY){
				await this.loadHierarchy(node);
			}

			let workerPath = "./src/potree/DecoderWorker_copc.js";
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
			let url = new URL(this.url, document.baseURI).href;
			let pointAttributes = this.attributes;
			let scale = this.header.scale;
			let offset = this.header.offset;
			let min = this.octree.loader.header.min;
			let numPoints = node.numPoints;
			let name = node.name;
			let nodeMin = [
				node.boundingBox.min.x,// + min[0],
				node.boundingBox.min.y,// + min[1],
				node.boundingBox.min.z,// + min[2],
			];
			let nodeMax = [
				node.boundingBox.max.x,// + min[0],
				node.boundingBox.max.y,// + min[1],
				node.boundingBox.max.z,// + min[2],
			];

			let pointFormat = this.header.pointFormat % 128;
			let pointRecordLength = this.header.pointRecordLength;

			let message = {
				name, url, byteOffset, byteSize, numPoints,
				pointAttributes, scale, offset, min, nodeMin, nodeMax,
				pointFormat, pointRecordLength,
			};

			worker.postMessage(message, []);
		}catch(e){
			node.loaded = false;
			node.loading = false;
			nodesLoading--;

			console.log(`failed to load ${node.name}`);
			console.log(e);
			console.log(`trying again!`);
		}
	}

	static async load(url){

		let loader = new CopcLoader();
		loader.url = url;

		let header = await loadHeader(url);
		let vlrs = await loadVLRs(url, header);
		let copcInfo = parseCopcInfo(vlrs);

		if(!copcInfo){
			console.error("No CopcInfo VLR found?");
			return null;
		}

		loader.header = header;
		loader.vlrs = vlrs;
		loader.copcInfo = copcInfo;

		let octree = new PointCloudOctree();
		octree.url = url;
		octree.spacing = copcInfo.spacing;
		let min = copcInfo.center.clone().subScalar(copcInfo.halfsize);
		let max = copcInfo.center.clone().addScalar(copcInfo.halfsize);
		octree.boundingBox = new Box3(min, max);
		octree.position.copy(octree.boundingBox.min);
		octree.boundingBox.max.sub(octree.boundingBox.min);
		octree.boundingBox.min.set(0, 0, 0);
		octree.updateWorld();

		// let aXYZ = new PointAttribute("position", typenameTypeattributeMap["float"], 3);
		// let aRGB = new PointAttribute("rgba", typenameTypeattributeMap["uint16"], 3);
		// let attributes = new PointAttributes();
		// attributes.add(aXYZ);
		// attributes.add(aRGB);
		let attributesList = parseAttributes(header, vlrs);
		let attributes = new PointAttributes();
		for(let attribute of attributesList){
			attributes.add(attribute);
		}
		loader.attributes = attributes;

		octree.attributes = attributes;
		octree.loader = loader;
		loader.octree = octree;

		let root = new PointCloudOctreeNode("r");
		root.boundingBox = octree.boundingBox.clone();
		root.level = 0;
		root.nodeType = NodeType.PROXY;
		root.hierarchyByteOffset = copcInfo.root_hier_offset;
		root.hierarchyByteSize = copcInfo.root_hier_size;
		root.byteOffset = 0;
		root.spacing = octree.spacing;
		root.octree = octree;

		loader.loadNode(root);

		octree.root = root;

		Potree.events.dispatcher.dispatch("pointcloud_loaded", octree);

		return octree;
	}

}