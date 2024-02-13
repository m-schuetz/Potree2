import {loadVoxels} from "./loadVoxels.js";
import {loadPoints, loadPointsBrotli} from "./loadPoints.js";

// - voxels are encoded relative to parent.
// - the children of this chunk's root need the parent voxels.
// - <parentVoxelCoords> stores that
// - note that the root of this chunk also needs parent voxels, 
//   but these are passed from the main thread via postMessage
let parentVoxelCoords = null;

function loadNode(octree, node, dataview){

	if(node.numVoxels > 0){
		return loadVoxels(octree, node, dataview, parentVoxelCoords);
	}else if(node.numPoints > 0){
		// debugger;
		return loadPointsBrotli(octree, node, dataview);
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

	chunkOffset += event.data.metadata.pointBuffer.offset;

	let numElements = nodes.reduce( (sum, node) => sum + node.numPoints + node.numVoxels, 0);
	let bitsPerElement = Math.ceil(8 * chunkSize / numElements);

	// let strChunkSize = chunkSize.toLocaleString().padStart(10);
	// let strNumElements = numElements.toLocaleString().padStart(8);
	// let strBpe = bitsPerElement.toLocaleString().padStart(4);
	// let strBytes = (bitsPerElement / 8).toFixed(1).padStart(4);
	// console.log(`#nodes: ${nodes.length}, chunkSize: ${strChunkSize}, numElements: ${strNumElements}, bpe: ${strBpe} (${strBytes} bytes)`);

	let response = await fetch(url, {
		headers: {
			'content-type': 'multipart/byteranges',
			'Range': `bytes=${chunkOffset}-${chunkOffset + chunkSize - 1}`,
		},
	});

	// console.log(`loading chunk offset ${chunkOffset.toLocaleString()}. size ${chunkSize.toLocaleString()}`);

	let buffer = await response.arrayBuffer();

	parentVoxelCoords = event.data.parentVoxelCoords;

	for(let node of nodes){
		// debugger;
		let dataview = new DataView(buffer, 
			node.byteOffset - chunkOffset + event.data.metadata.pointBuffer.offset, 
			node.byteSize);
		
		let buffers = loadNode(octree, node, dataview);

		// clone voxel coords, the child nodes need them to decode their coords
		// if(node === nodes[0] && node.numVoxels > 0){
		// 	let voxelCoords = buffers.voxelCoords;
		// 	parentVoxelCoords = new Uint8Array(voxelCoords.byteLength);
		// 	parentVoxelCoords.set(voxelCoords);
		// }

		// done loading node, send results to main thread
		let message = {
			type: "node parsed",
			name: node.name,
			buffer: buffers.buffer,
			voxelCoords: buffers.voxelCoords
		};
		let transferables = [message.buffer];

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
		console.log(e);
		postMessage("failed");
	});

	
};
