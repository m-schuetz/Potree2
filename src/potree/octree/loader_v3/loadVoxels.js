
import {Line3} from "../../../math/Line3.js";
import {Vector3} from "../../../math/Vector3.js";

const gridSize = 128;

// round up to nearest <n>
function ceil_n(number, n){
	return number + (n - (number % n));
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

	let targetBuffer       = new ArrayBuffer(ceil_n(bytesPerPoint * numVoxels, 4));
	let target_coordinates = new DataView(targetBuffer, 0, 12 * numVoxels);
	let target_rgb         = new DataView(targetBuffer, offset_rgb * numVoxels, 6 * numVoxels);
	let voxelCoords        = new Uint8Array(3 * numVoxels);

	let nodeSize = [
		node.max[0] - node.min[0],
		node.max[1] - node.min[1],
		node.max[2] - node.min[2],
	];

	if(node.name === "r"){
		// root node encodes voxel coordinates directly

		for(let i = 0; i < numVoxels; i++){
			let cx = source.getUint8(3 * i + 0) + 0.5;
			let cy = source.getUint8(3 * i + 1) + 0.5;
			let cz = source.getUint8(3 * i + 2) + 0.5;

			let x = (cx / gridSize) * nodeSize[0] + node.min[0];
			let y = (cy / gridSize) * nodeSize[1] + node.min[1];
			let z = (cz / gridSize) * nodeSize[2] + node.min[2];

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

			let r = source.getUint8(3 * numVoxels + 3 * i + 0);
			let g = source.getUint8(3 * numVoxels + 3 * i + 1);
			let b = source.getUint8(3 * numVoxels + 3 * i + 2);
			target_rgb.setUint16(6 * i + 0, r, true);
			target_rgb.setUint16(6 * i + 2, g, true);
			target_rgb.setUint16(6 * i + 4, b, true);
		}

	}else{
		// other inner nodes encode voxels relative to parent voxels

		let parentVoxels = parentVoxelCoords;
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

					voxelCoords[3 * numGeneratedVoxels + 0] = ix;
					voxelCoords[3 * numGeneratedVoxels + 1] = iy;
					voxelCoords[3 * numGeneratedVoxels + 2] = iz;

					let x = nodeSize[0] * ((ix + 0.5) / gridSize) + node.min[0];
					let y = nodeSize[1] * ((iy + 0.5) / gridSize) + node.min[1];
					let z = nodeSize[2] * ((iz + 0.5) / gridSize) + node.min[2];

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


		// normal rgb decoding
		// let numChildmasks = i;
		// let rgbOffset = numChildmasks;
		// for(let i = 0; i < numGeneratedVoxels; i++){
		// 	let r = source.getUint8(rgbOffset + 3 * i + 0);
		// 	let g = source.getUint8(rgbOffset + 3 * i + 1);
		// 	let b = source.getUint8(rgbOffset + 3 * i + 2);
		// 	target_rgb.setUint16(6 * i + 0, r, true);
		// 	target_rgb.setUint16(6 * i + 2, g, true);
		// 	target_rgb.setUint16(6 * i + 4, b, true);
		// }

		// BC-ish decoding
		let numChildmasks = i;
		let rgbOffset = numChildmasks;
		let rgbByteSize = node.byteSize - rgbOffset;
		let blocksize = 8;
		let bytesPerBlock = 8;
		let bitsPerSample = 2;
		let numSamples = 4;
		let numBlocks = rgbByteSize / blocksize;

		let colorsProcessed = 0;
		let color = new Vector3();
		let line = new Line3();
		for(let blockIndex = 0; blockIndex < numBlocks; blockIndex++){

			line.start.x = source.getUint8(rgbOffset + bytesPerBlock * blockIndex + 0);
			line.start.y = source.getUint8(rgbOffset + bytesPerBlock * blockIndex + 1);
			line.start.z = source.getUint8(rgbOffset + bytesPerBlock * blockIndex + 2);
			line.end.x   = source.getUint8(rgbOffset + bytesPerBlock * blockIndex + 3);
			line.end.y   = source.getUint8(rgbOffset + bytesPerBlock * blockIndex + 4);
			line.end.z   = source.getUint8(rgbOffset + bytesPerBlock * blockIndex + 5);

			let bits = source.getUint16(rgbOffset + bytesPerBlock * blockIndex + 6, true);

			for(let sampleIndex = 0; sampleIndex < blocksize; sampleIndex++){

				let T = bits >>> (bitsPerSample * sampleIndex) & 0b11;
				let t = T / (numSamples - 1);

				line.at(t, color);

				target_rgb.setUint16(6 * colorsProcessed + 0, color.x, true);
				target_rgb.setUint16(6 * colorsProcessed + 2, color.y, true);
				target_rgb.setUint16(6 * colorsProcessed + 4, color.z, true);

				colorsProcessed++;
				if(colorsProcessed === numVoxels) break;
			}
			if(colorsProcessed === numVoxels) break;
		}
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

	return {buffer: targetBuffer, voxelCoords};
}