import {loadVoxels} from "./loadVoxels.js";
import {loadPoints} from "./loadPoints.js";

// - voxels are encoded relative to parent.
// - the children of this chunk's root need the parent voxels.
// - <parentVoxelCoords> stores that
// - note that the root of this chunk also needs parent voxels, 
//   but these are passed from the main thread via postMessage
let parentVoxelCoords = null;

function loadNode(octree, node, dataview){

	// console.log(`loading ${node.name}`);

	if(node.numVoxels > 0){
		return loadVoxels(octree, node, dataview, parentVoxelCoords);
	}else if(node.numPoints > 0){
		return loadPoints(octree, node, dataview);
	}

}

async function loadNodes(event){

	let {octree, nodes, url} = event.data;

	let chunkOffset = Infinity;
	let chunkSize = 0;

	let byLevel = (a, b) => a.level - b.level;
	nodes.sort(byLevel);

	for(let node of nodes){
		chunkOffset = Math.min(chunkOffset, node.byteOffset);
		chunkSize += node.byteSize;
	}

	let numElements = nodes.reduce( (sum, node) => sum + node.numPoints + node.numVoxels, 0);
	let bitsPerElement = Math.ceil(8 * chunkSize / numElements);
	console.log(`#nodes: ${nodes.length}, chunkSize: ${chunkSize}, numElements: ${numElements}, bpe: ${bitsPerElement}`);

	let response = await fetch(url, {
		headers: {
			'content-type': 'multipart/byteranges',
			'Range': `bytes=${chunkOffset}-${chunkOffset + chunkSize - 1}`,
		},
	});

	let buffer = await response.arrayBuffer();

	parentVoxelCoords = event.data.parentVoxelCoords;

	for(let node of nodes){
		let dataview = new DataView(buffer, node.byteOffset - chunkOffset, node.byteSize);
		
		let buffers = loadNode(octree, node, dataview);

		// first = chunk root
		// clone voxel coords, the child nodes need them to decode their coords
		if(node === nodes[0] && node.numVoxels > 0){
			let voxelCoords = buffers.voxelCoords;
			parentVoxelCoords = new Uint8Array(voxelCoords.byteLength);
			parentVoxelCoords.set(voxelCoords);
		}

		// done loading node, send results to main thread
		let message = {
			type: "node parsed",
			name: node.name,
			buffer: buffers.buffer,
			voxelCoords: buffers.voxelCoords
		};
		let transferables = [buffers.buffer];

		if(buffers.voxelCoords){
			transferables.push(buffers.voxelCoords.buffer);
		}

		postMessage(message, transferables);
	}

}

onmessage = async function (event) {

	let promise = loadNodes(event);

	promise.then(e => {
		postMessage("finished");
	});

	// Chrome frequently fails with range requests.
	// Notify main thread that loading failed, so that it can try again.
	promise.catch(e => {
		debugger;
		console.log(e);
		postMessage("failed");
	});

	
};
