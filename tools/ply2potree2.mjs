
import {promises as fsp} from "fs";

// let plyDir = "D:/temp/wetransfer-05965b/Additive-Heidentor-Non-Averaged";
// let targetDir = "D:/temp/wetransfer-05965b/converted_additive_no_avg";

// let plyDir = "D:/temp/wetransfer-05965b/Additive-Heidentor-Averaged";
// let targetDir = "D:/temp/wetransfer-05965b/converted_additive_avg";

let plyDir = "D:/temp/wetransfer-05965b/Replacement-Heidentor-Averaged";
let targetDir = "D:/temp/wetransfer-05965b/converted_replacing_avg";

let p_offset = [0, 0, 0]; // shift coordinates by that value before integer conversion
let p_scale = 0.001; // integer conversion scale. 0.001 = millimeter precision

let boundingBox = {
	min: [Infinity, Infinity, Infinity],
	max: [-Infinity, -Infinity, -Infinity],
};

// let boundingBox = {
// 	min: [-8.096, -4.800, 1.687],
// 	max: [0.453, 10.758, 15.803],
// };

// let boxSize = Math.max(
// 	boundingBox.max[0] - boundingBox.min[0],
// 	boundingBox.max[1] - boundingBox.min[1],
// 	boundingBox.max[2] - boundingBox.min[2],
// );
// boundingBox.max[0] = boundingBox.min[0] + boxSize;
// boundingBox.max[1] = boundingBox.min[1] + boxSize;
// boundingBox.max[2] = boundingBox.min[2] + boxSize;

// p_offset = boundingBox.min;



function expandBox(x, y, z){
	boundingBox.min[0] = Math.min(boundingBox.min[0], x);
	boundingBox.min[1] = Math.min(boundingBox.min[1], y);
	boundingBox.min[2] = Math.min(boundingBox.min[2], z);

	boundingBox.max[0] = Math.max(boundingBox.max[0], x);
	boundingBox.max[1] = Math.max(boundingBox.max[1], y);
	boundingBox.max[2] = Math.max(boundingBox.max[2], z);
}


class Node{

	constructor(name){
		this.name = name;
		this.children = new Array(8).fill(null);
		this.byteOffset = 0;
		this.byteSize = 0;
	}

	toChildMask(){

		let mask = 0;

		for(let i = 0; i < 8; i++){
			if(this.children[i] != null){
				mask = mask | (1 << i);
			}
		}

		return mask;
	}

}

function buildHierarchy(files){

	files.sort((a, b) => {
		if(a.length !== b.length){
			return a.length - b.length;
		}else{
			return 0;
		}
	});

	let nodes = [];
	let root = new Node("r");
	let map = new Map();
	map.set("r", root);
	nodes.push(root);

	for(let i = 1; i < files.length; i++){
		let file = files[i];
		let name = file.substring(0, file.length - 4);
		let parentName = name.substring(0, name.length - 1);

		let childIndex = parseInt(name[name.length - 1]);
		let parent = map.get(parentName);
		let child = new Node(name);

		parent.children[childIndex] = child;
		map.set(name, child);
		nodes.push(child);
	}

	return {root, nodes};
}

function toCharCodeArray(string){
	let array = [];

	for(let i = 0; i < string.length; i++){
		array.push(string.charCodeAt(i));
	}

	return array;
}

function getBinaryContentStart(buffer){
	let strEndHeader = "end_header";
	let endHeaderCode = toCharCodeArray("end_header");
	let u8 = new Uint8Array(buffer);

	let j = 0;
	for(let i = 0; i < buffer.length; i++){

		if(u8[i] === endHeaderCode[j]){
			j++;
		}else{
			j = 0;
		}

		if(j === endHeaderCode.length - 1){
			
			// +1 to jump to "r"
			// +1 to jump to "\n"
			// +1 to jump to first byte after "end_header\n"
			// assumes that the line feed is \r without any \r
			return i + 3;
		}
	}

	throw "no 'end_header' found";
}

async function nodeToBuffer(node){
	let path = `${plyDir}/${node.name}.ply`;

	let buffer = await fsp.readFile(path);
	let start = getBinaryContentStart(buffer);

	let size = buffer.length - start;
	let numPoints = size / 15;
	node.numPoints = numPoints;

	// 4 * 3 coordinates, 3 * 2 colors
	let source = buffer;
	let target = Buffer.alloc(numPoints * 18);
	// let source = new DataView(buffer.buffer);
	// let out = new ArrayBuffer(numPoints * 18);
	// let target = new DataView(out);

	for(let i = 0; i < numPoints; i++){

		{ // XYZ
			let x = source.readFloatLE(start + i * 15 + 0);
			let y = source.readFloatLE(start + i * 15 + 4);
			let z = source.readFloatLE(start + i * 15 + 8);

			expandBox(x, y, z);

			let X = Math.floor((x - p_offset[0]) / p_scale);
			let Y = Math.floor((y - p_offset[1]) / p_scale);
			let Z = Math.floor((z - p_offset[2]) / p_scale);

			target.writeInt32LE(X, i * 18 + 0);
			target.writeInt32LE(Y, i * 18 + 4);
			target.writeInt32LE(Z, i * 18 + 8);
		}

		{ // RGB
			let r = source.readUInt8(start + i * 15 + 12);
			let g = source.readUInt8(start + i * 15 + 13);
			let b = source.readUInt8(start + i * 15 + 14);

			target.writeUInt16LE(r, i * 18 + 12);
			target.writeUInt16LE(g, i * 18 + 14);
			target.writeUInt16LE(b, i * 18 + 16);
		}

	}

	return target;
}

async function createPointFile(nodes){

	let filepath = `${targetDir}/octree.bin`;
	try{
		await fsp.unlink(filepath);
	}catch(e){}

	let outfile = await fsp.open(filepath, "a");

	let bytesWritten = 0;
	for(let node of nodes){
		
		let buffer = await nodeToBuffer(node);

		node.byteOffset = bytesWritten;
		node.byteSize = buffer.length;

		await fsp.appendFile(outfile, buffer);
		bytesWritten += buffer.length;
	}

}

function traverse(node, callback){

	callback(node);

	for(let child of node.children){
		if(child != null){
			traverse(child, callback);
		}
	}

}

async function createHierarchyFile(root, nodes){

	// sort breadth-first
	nodes.sort((a, b) => {
		if(a.name.length !== b.name.length){
			return a.name.length - b.name.length;
		}else{

			return parseInt(a.name.substring(1)) - parseInt(b.name.substring(1));
		}
	});

	let bytesPerNode = 22;
	let buffer = Buffer.alloc(nodes.length * bytesPerNode);

	for(let i = 0; i < nodes.length; i++){
		let node = nodes[i];

		let childMask = node.toChildMask();
		let type = childMask === 0 ? 1 : 0;

		buffer.writeUInt8(type, i * bytesPerNode + 0);
		buffer.writeUInt8(childMask, i * bytesPerNode + 1);
		buffer.writeUInt32LE(node.numPoints, i * bytesPerNode + 2);
		buffer.writeBigUInt64LE(BigInt(node.byteOffset), i * bytesPerNode + 6);
		buffer.writeBigUInt64LE(BigInt(node.byteSize) , i * bytesPerNode + 14);
	}

	await fsp.writeFile(`${targetDir}/hierarchy.bin`, buffer);
}

async function createMetadataFile(root, nodes){

	let numPoints = 0;
	traverse(root, (node) => {
		numPoints += node.numPoints;
	});

	let boxSize = Math.max(
		boundingBox.max[0] - boundingBox.min[0],
		boundingBox.max[1] - boundingBox.min[1],
		boundingBox.max[2] - boundingBox.min[2],
	);
	boundingBox.max[0] = boundingBox.min[0] + boxSize;
	boundingBox.max[1] = boundingBox.min[1] + boxSize;
	boundingBox.max[2] = boundingBox.min[2] + boxSize;

	let spacing = boxSize / 128;

	let metadata = {
		version: "2.0",
		name: "abc",
		description: "",
		points: numPoints,
		projection: "",
		hierarchy: {
			firstChunkSize: nodes.length * 22,
			stepSize: 100,
			depth: 20,
		},
		offset: boundingBox.min,
		scale: [p_scale, p_scale, p_scale],
		spacing: spacing,
		boundingBox: boundingBox,
		encoding: "DEFAULT",
		attributes: [
			{
				"name": "position",
				"description": "",
				"size": 12,
				"numElements": 3,
				"elementSize": 4,
				"type": "int32",
				"min": boundingBox.min,
				"max": boundingBox.max,
			},{
				"name": "rgb",
				"description": "",
				"size": 6,
				"numElements": 3,
				"elementSize": 2,
				"type": "uint16",
				"min": [0, 0, 0],
				"max": [65024, 65280, 65280]
			}
		],
	};

	let strJson = JSON.stringify(metadata, null, '\t');

	await fsp.writeFile(`${targetDir}/metadata.json`, strJson);

}

async function convert(){

	try{
		await fsp.mkdir(targetDir);
	}catch(e){}

	let listing = await fsp.readdir(plyDir);
	let plyFiles = listing.filter(file => file.endsWith(".ply"));
	plyFiles = plyFiles.filter(file => file.length <= 9);

	let {root, nodes} = buildHierarchy(plyFiles);

	await createPointFile(nodes);

	await createHierarchyFile(root, nodes);
	await createMetadataFile(root, nodes);

	console.log(root);
}



convert();
