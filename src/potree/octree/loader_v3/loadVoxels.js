
import {Line3} from "../../../math/Line3.js";
import {Vector3} from "../../../math/Vector3.js";

const gridSize = 128;

// round up to nearest <n>
function ceil_n(number, n){

	if(number % n === 0){
		return number;
	}else{
		return number - (number % n) + n;
	}
}

function loadBC(source, target, numVoxels){

	let blocksize = 8;
	let bytesPerBlock = 8;
	let bitsPerSample = 2;
	let numSamples = 4;
	let numBlocks = Math.ceil(numVoxels / blocksize);

	let colorsProcessed = 0;
	let color = new Vector3();
	let line = new Line3();
	for(let blockIndex = 0; blockIndex < numBlocks; blockIndex++){

		line.start.x = source.getUint8(bytesPerBlock * blockIndex + 0);
		line.start.y = source.getUint8(bytesPerBlock * blockIndex + 1);
		line.start.z = source.getUint8(bytesPerBlock * blockIndex + 2);
		line.end.x   = source.getUint8(bytesPerBlock * blockIndex + 3);
		line.end.y   = source.getUint8(bytesPerBlock * blockIndex + 4);
		line.end.z   = source.getUint8(bytesPerBlock * blockIndex + 5);

		let bits = source.getUint16(bytesPerBlock * blockIndex + 6, true);

		// if(blockIndex === 0){
		// 	debugger;
		// }

		for(let sampleIndex = 0; sampleIndex < blocksize; sampleIndex++){

			let T = bits >>> (bitsPerSample * sampleIndex) & 0b11;
			let t = T / (numSamples - 1);

			line.at(t, color);

			target.setUint16(6 * colorsProcessed + 0, color.x, true);
			target.setUint16(6 * colorsProcessed + 2, color.y, true);
			target.setUint16(6 * colorsProcessed + 4, color.z, true);

			colorsProcessed++;
			if(colorsProcessed === numVoxels) break;
		}
		if(colorsProcessed === numVoxels) break;
	}

}


export function loadVoxels(octree, node, source, parentVoxelCoords){

	let tStart = performance.now();
	let tStartParse = performance.now();

	let {numVoxels} = node;

	// includes position+filtered+unfiltered attributes
	// unfiltered are loaded at a later time
	let bytesPerPoint = octree.pointAttributes.byteSize;

	let offset_rgb = 0;
	for(let attribute of octree.pointAttributes.attributes){
		if(["rgb", "rgba"].includes(attribute.name)){
			offset_rgb = attribute.byteOffset;
			break;
		}
	}

	// let targetBuffer       = new ArrayBuffer(ceil_n(bytesPerPoint * numVoxels, 4));
	// let target_coordinates = new DataView(targetBuffer, 0, 12 * numVoxels);
	// let target_rgb         = new DataView(targetBuffer, offset_rgb * numVoxels, 6 * numVoxels);
	// let voxelCoords        = new Uint8Array(3 * numVoxels);

	// 3 byte xyz, 1 byte BC-ish encoded colors
	let xyzByteSize = 3 * numVoxels;
	let rgbByteSize = ceil_n(numVoxels, 8);
	let numBytes = ceil_n(xyzByteSize + rgbByteSize, 16);
	// let numBytes = ceil_n(3 * numVoxels + numVoxels, 16);
	let targetBuffer    = new ArrayBuffer(numBytes);
	let target_xyz      = new DataView(targetBuffer, 0, 3 * numVoxels);
	let target_rgb      = new DataView(targetBuffer, 3 * numVoxels, ceil_n(numVoxels, 8));

	// if(node.name === "r444"){
	// }
	// debugger;

	if(node.name === "r"){
		// root node encodes voxel coordinates directly

		let rgbOffset = 3 * numVoxels;

		for(let i = 0; i < numVoxels; i++){
			target_xyz.setUint8(3 * i + 0, source.getUint8(3 * i + 0));
			target_xyz.setUint8(3 * i + 1, source.getUint8(3 * i + 1));
			target_xyz.setUint8(3 * i + 2, source.getUint8(3 * i + 2));
			target_rgb.setUint8(i, source.getUint8(rgbOffset + i));
		}

	}else{
		// other inner nodes encode voxels relative to parent voxels

		

		let parentVoxels = new Uint8Array(parentVoxelCoords);
		let numParentVoxels = parentVoxels.length / 3;
		let thisChildIndex = parseInt(node.name.at(node.name.length - 1));

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
		// debugger;
		let numGeneratedVoxels = 0;
		let i = 0;
		while(numGeneratedVoxels < numVoxels){

			let childmask = source.getUint8(i);
			let px = parentVoxels[3 * parent_i + 0];
			let py = parentVoxels[3 * parent_i + 1];
			let pz = parentVoxels[3 * parent_i + 2];

			for(let ci = 0; ci < 8; ci++){
				if(((childmask >>> ci) & 1) !== 0){
					// found valid child voxel

					let cx = (ci >>> 2) & 1;
					let cy = (ci >>> 1) & 1;
					let cz = (ci >>> 0) & 1;

					let ix = 2 * (px % (gridSize / 2)) + cx;
					let iy = 2 * (py % (gridSize / 2)) + cy;
					let iz = 2 * (pz % (gridSize / 2)) + cz;

					// voxelCoords[3 * numGeneratedVoxels + 0] = ix;
					// voxelCoords[3 * numGeneratedVoxels + 1] = iy;
					// voxelCoords[3 * numGeneratedVoxels + 2] = iz;

					target_xyz.setUint8(3 * numGeneratedVoxels + 0, ix);
					target_xyz.setUint8(3 * numGeneratedVoxels + 1, iy);
					target_xyz.setUint8(3 * numGeneratedVoxels + 2, iz);

					// let x = nodeSize[0] * ((ix + 0.5) / gridSize) + node.min[0];
					// let y = nodeSize[1] * ((iy + 0.5) / gridSize) + node.min[1];
					// let z = nodeSize[2] * ((iz + 0.5) / gridSize) + node.min[2];

					// if(12 * numGeneratedVoxels + 8 > target_coordinates.byteLength){
					// 	debugger;
					// }
					// target_coordinates.setFloat32(12 * numGeneratedVoxels + 0, x, true);
					// target_coordinates.setFloat32(12 * numGeneratedVoxels + 4, y, true);
					// target_coordinates.setFloat32(12 * numGeneratedVoxels + 8, z, true);

					numGeneratedVoxels++;
				}
			}

			i++;
			parent_i++;
		}

		// normal rgb decoding
		let numChildmasks = i;
		let rgbOffset = numChildmasks;

		// BC-ish decoding
		let source_bc  = new DataView(source.buffer, source.byteOffset + rgbOffset, ceil_n(numVoxels, 8));

		for(let i = 0; i < numVoxels; i++){
			target_rgb.setUint8(i, source_bc.getUint8(i));
		}

		// loadBC(source_bc, target_rgb, numVoxels);
	}

	// let dTotal = performance.now() - tStart;
	// let dParse = performance.now() - tStartParse;
	// // let dBrotli = tEndBrotliDecode - tStartBrotliDecode;
	// let mps = (1000 * numVoxels / dParse) / 1_000_000;

	// let strDTotal = dTotal.toFixed(1).padStart(4);
	// let strDParse = dParse.toFixed(1).padStart(4);
	// // let strDBrotli = dBrotli.toFixed(1).padStart(4);
	// let strMPS = mps.toFixed(1).padStart(5);
	// let strVoxels = (numVoxels + "").padStart(5);
	// let strKB = (Math.floor(buffer.byteLength / 1024) + "").padStart(4);

	// console.log(`[${name.padStart(10)}] #voxels: ${strVoxels}, ${strKB} kb, parse: ${strDParse} ms, total: ${strDTotal} ms. ${strMPS} MP/s`);
	// console.log(`[${name.padStart(10)}] #voxels: ${strVoxels}, ${strKB} kb, brotli: ${strDBrotli} ms, parse: ${strDParse} ms, total: ${strDTotal} ms. ${strMPS} MP/s`);

	return {buffer: targetBuffer};
}