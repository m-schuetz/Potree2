
// import {Box3, Vector3} from "potree";
import {BrotliDecode} from "../../../../libs/brotli/decode.js";

const gridSize = 128;

class Stats{
	constructor(){
		this.name = "";
		this.min = null;
		this.max = null;
		this.mean = null;
	}
};

function getBoxSize(min, max){
	let boxSize = [
		max[0] - min[0],
		max[1] - min[1],
		max[2] - min[2],
	];

	return boxSize;
}

// round up to nearest <n>
function ceil_n(number, n){
	return number + (n - (number % n));
}

async function loadNode(event){

	let tStart = performance.now();

	let data = event.data;
	let numVoxels = data.numElements;
	let name = data.name;

	let first = data.byteOffset;
	let last = first + data.byteSize - 1;
	// let last = first + Math.min(numVoxels * 6, data.byteSize);

	let response = await fetch(data.url, {
		headers: {
			'content-type': 'multipart/byteranges',
			'Range': `bytes=${first}-${last}`,
		},
	});

	let buffer = await response.arrayBuffer();

	// let tStartBrotliDecode = performance.now();
	// buffer = BrotliDecode(new Int8Array(buffer)).buffer;
	// let tEndBrotliDecode = performance.now();

	let source = new DataView(buffer, 0);

	let tStartParse = performance.now();

	let targetBuffer = new ArrayBuffer(ceil_n(18 * numVoxels, 4));
	let target_coordinates = new DataView(targetBuffer, 0, 12 * numVoxels);
	let target_rgb = new DataView(targetBuffer, 12 * numVoxels, 6 * numVoxels);
	let voxelCoords = new Uint8Array(3 * numVoxels);

	let nodeMin = data.nodeMin;
	let nodeMax = data.nodeMax;

	let nodeSize = [
		nodeMax[0] - nodeMin[0],
		nodeMax[1] - nodeMin[1],
		nodeMax[2] - nodeMin[2],
	];

	if(data.name === "r"){
		// root node encodes voxel coordinates directly

		for(let i = 0; i < numVoxels; i++){
			let cx = source.getUint8(3 * i + 0) + 0.5;
			let cy = source.getUint8(3 * i + 1) + 0.5;
			let cz = source.getUint8(3 * i + 2) + 0.5;

			let x = (cx / gridSize) * nodeSize[0] + nodeMin[0];
			let y = (cy / gridSize) * nodeSize[1] + nodeMin[1];
			let z = (cz / gridSize) * nodeSize[2] + nodeMin[2];

			voxelCoords[3 * i + 0] = source.getUint8(3 * i + 0);
			voxelCoords[3 * i + 1] = source.getUint8(3 * i + 1);
			voxelCoords[3 * i + 2] = source.getUint8(3 * i + 2);

			target_coordinates.setFloat32(12 * i + 0, x, true);
			target_coordinates.setFloat32(12 * i + 4, y, true);
			target_coordinates.setFloat32(12 * i + 8, z, true);

			let mortoncode = i;
				
			let mx = 0;
			let my = 0;
			for(let bitindex = 0; bitindex < 10; bitindex++){
				let bx = (mortoncode >> (2 * bitindex + 0)) & 1;
				let by = (mortoncode >> (2 * bitindex + 1)) & 1;

				mx = mx | (bx << bitindex);
				my = my | (by << bitindex);
			}

			// target_rgb.setUint16(6 * i + 0, source.getUint8(3 * numVoxels + 3 * i + 0), true);
			// target_rgb.setUint16(6 * i + 2, source.getUint8(3 * numVoxels + 3 * i + 1), true);
			// target_rgb.setUint16(6 * i + 4, source.getUint8(3 * numVoxels + 3 * i + 2), true);
			let r = source.getUint8(3 * numVoxels + 3 * i + 0);
			let g = source.getUint8(3 * numVoxels + 3 * i + 1);
			let b = source.getUint8(3 * numVoxels + 3 * i + 2);
			target_rgb.setUint16(6 * i + 0, r, true);
			target_rgb.setUint16(6 * i + 2, g, true);
			target_rgb.setUint16(6 * i + 4, b, true);
		}

	}else{
		// other inner nodes encode voxels relative to parent voxels

		let parentVoxels = event.data.parentVoxelCoords;
		let numParentVoxels = parentVoxels.length / 3;
		let thisChildIndex = parseInt(name.at(name.length - 1));

		// debugger;

		// find first parent voxel of current node's octant
		let parent_i = 0;
		for(; parent_i < numParentVoxels; parent_i++){

			let p_x = parentVoxels[3 * parent_i + 0];
			let p_y = parentVoxels[3 * parent_i + 1];
			let p_z = parentVoxels[3 * parent_i + 2];

			let cx = (p_x < gridSize / 2) ? 0 : 1;
			let cy = (p_y < gridSize / 2) ? 0 : 1;
			let cz = (p_z < gridSize / 2) ? 0 : 1;

			let childIndex = (cx << 2) | (cy << 1) | cz;

			if(childIndex === thisChildIndex) break;
		}

		// now parent_i points to first parent voxel inside current node
		// next, use child masks to break parent voxels into current node's voxels

		let numGeneratedVoxels = 0;
		let i = 0;
		while(numGeneratedVoxels < numVoxels){

			let childmask = source.getUint8(i);
			let px = parentVoxels[3 * parent_i + 0];
			let py = parentVoxels[3 * parent_i + 1];
			let pz = parentVoxels[3 * parent_i + 2];

//######### METHOD 1
			for(let ci = 0; ci < 8; ci++){
				if(((childmask >>> ci) & 1) !== 0){
					// found valid child voxel

//######### METHOD 2
			// for(let ci of [0, 2, 4, 6, 1, 3, 5, 7]){
			// 	if(((childmask >>> ci) & 1) !== 0){
					// found valid child voxel

//######### METHOD 3
			// for(let cz of [0, 1])
			// for(let cx of [0, 1])
			// for(let cy of [0, 1])
			// {
			// 	let ci = (cx << 2) | (cy << 1) | cz;
			// 	if(((childmask >>> ci) & 1) !== 0){
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
					let cx = (ci >>> 2) & 1;
					let cy = (ci >>> 1) & 1;
					let cz = (ci >>> 0) & 1;

					let ix = 2 * (px % (gridSize / 2)) + cx;
					let iy = 2 * (py % (gridSize / 2)) + cy;
					let iz = 2 * (pz % (gridSize / 2)) + cz;

					voxelCoords[3 * numGeneratedVoxels + 0] = ix;
					voxelCoords[3 * numGeneratedVoxels + 1] = iy;
					voxelCoords[3 * numGeneratedVoxels + 2] = iz;

					let x = nodeSize[0] * ((ix + 0.5) / gridSize) + nodeMin[0];
					let y = nodeSize[1] * ((iy + 0.5) / gridSize) + nodeMin[1];
					let z = nodeSize[2] * ((iz + 0.5) / gridSize) + nodeMin[2];

					if(12 * numGeneratedVoxels + 8 > target_coordinates.byteLength){
						debugger;
					}
					target_coordinates.setFloat32(12 * numGeneratedVoxels + 0, x, true);
					target_coordinates.setFloat32(12 * numGeneratedVoxels + 4, y, true);
					target_coordinates.setFloat32(12 * numGeneratedVoxels + 8, z, true);

					numGeneratedVoxels++;
				}
			}

			i++;
			parent_i++;
		}

		let numChildmasks = i;
		let rgbOffset = numChildmasks;
		for(let i = 0; i < numGeneratedVoxels; i++){
			let r = source.getUint8(rgbOffset + 3 * i + 0);
			let g = source.getUint8(rgbOffset + 3 * i + 1);
			let b = source.getUint8(rgbOffset + 3 * i + 2);
			target_rgb.setUint16(6 * i + 0, r, true);
			target_rgb.setUint16(6 * i + 2, g, true);
			target_rgb.setUint16(6 * i + 4, b, true);
		}
	}

	let dTotal = performance.now() - tStart;
	let dParse = performance.now() - tStartParse;
	// let dBrotli = tEndBrotliDecode - tStartBrotliDecode;
	let mps = (1000 * numVoxels / dParse) / 1_000_000;

	let strDTotal = dTotal.toFixed(1).padStart(4);
	let strDParse = dParse.toFixed(1).padStart(4);
	// let strDBrotli = dBrotli.toFixed(1).padStart(4);
	let strMPS = mps.toFixed(1).padStart(5);
	let strVoxels = (numVoxels + "").padStart(5);
	let strKB = (Math.floor(buffer.byteLength / 1024) + "").padStart(4);

	console.log(`[${name.padStart(10)}] #voxels: ${strVoxels}, ${strKB} kb, parse: ${strDParse} ms, total: ${strDTotal} ms. ${strMPS} MP/s`);
	// console.log(`[${name.padStart(10)}] #voxels: ${strVoxels}, ${strKB} kb, brotli: ${strDBrotli} ms, parse: ${strDParse} ms, total: ${strDTotal} ms. ${strMPS} MP/s`);

	let message = {
		type: "node parsed",
		buffer: targetBuffer,
		voxelCoords
	};
	let transferables = [targetBuffer, voxelCoords.buffer];

	postMessage(message, transferables);
}

onmessage = async function (event) {

	try{
		loadNode(event);
	}catch(e){
		console.log(e);
		postMessage("failed");
	}
	
};
