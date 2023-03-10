
// import {Box3, Vector3} from "potree";

import {BrotliDecode} from "../../../../libs/brotli/decode.js";

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

async function loadNode(node, nodeSpacing, parent, buffer){

	let tStart = performance.now();

	let view_voxel = new DataView(buffer, node.voxelBufferOffset);
	let view_jpeg = new DataView(buffer, node.jpegBufferOffset);
	let view_colBuffer = new DataView(buffer, node.colBufferOffset);

	let boxSize = getBoxSize(node.min, node.max);

	let target = new ArrayBuffer(node.numVoxels * 20);
	let view_target = new DataView(target);

	let offset_xyz = 0;
	let offset_rgb = 12 * node.numVoxels;

	// LOAD BROTLI COMPRESSED DIFF COLORS
	// let compressedBuffer = new Int8Array(buffer, node.colDiffCompressedBufferOffset,node.colDiffCompressedBufferSize);
	// let decoded = BrotliDecode(compressedBuffer);
	// let decodedView = new DataView(decoded.buffer);
	// view_colBuffer = decodedView;
	// let dDecompressColor = performance.now() - tStart;

	// LOAD BROTLI COMPRESSED COLORS
	// let compressedBuffer = new Int8Array(buffer, node.colCompressedBufferOffset,node.colCompressedBufferSize);
	// let decoded = BrotliDecode(compressedBuffer);
	// let decodedView = new DataView(decoded.buffer);
	// view_colBuffer = decodedView;
	// let dDecompressColor = performance.now() - tStart;

	// LOAD COLORS
	// let compressedBuffer = new Int8Array(buffer, node.colBufferOffset,node.colBufferSize);
	// // let decoded = BrotliDecode(compressedBuffer);
	// let decodedView = new DataView(decoded.buffer);
	// view_colBuffer = decodedView;
	// let dDecompressColor = performance.now() - tStart;



	// debugger;

	// LOAD JPEG ENCODED COLORS
	// let imageData;
	// { // try loading image
	// 	let blob = new Blob([new Uint8Array(buffer, node.jpegBufferOffset, node.jpegBufferSize)]);
	// 	let bitmap = await createImageBitmap(blob);

	// 	let [width, height] = [bitmap.width, bitmap.height];

	// 	const canvas = new OffscreenCanvas(256, 256);
	// 	const context = canvas.getContext("2d");
	// 	context.drawImage(bitmap, 0, 0);

	// 	imageData = context.getImageData(0, 0, width, height);
	// }
	// let dJpeg = performance.now() - tStart;

	// let readPixel = (x, y) => {
	// 	let pixelID = x + imageData.width * y;

	// 	let r = imageData.data[4 * pixelID + 0];
	// 	let g = imageData.data[4 * pixelID + 1];
	// 	let b = imageData.data[4 * pixelID + 2];
	// 	let a = imageData.data[4 * pixelID + 3];

	// 	return [r, g, b, a];
	// };


	let tVoxelStart = performance.now();
	if(node.name === "r"){
		// root node, directly encoded in voxel coordinates

		for(let i = 0; i < node.numVoxels; i++){
			let cx = view_voxel.getUint8(3 * i + 0) + 0.5;
			let cy = view_voxel.getUint8(3 * i + 1) + 0.5;
			let cz = view_voxel.getUint8(3 * i + 2) + 0.5;
			let x = (cx / 128.0) * boxSize[0] + node.min[0];
			let y = (cy / 128.0) * boxSize[1] + node.min[1];
			let z = (cz / 128.0) * boxSize[2] + node.min[2];

			view_target.setFloat32(12 * i + 0, x, true);
			view_target.setFloat32(12 * i + 4, y, true);
			view_target.setFloat32(12 * i + 8, z, true);

			let mortoncode = i;
				
			let mx = 0;
			let my = 0;
			for(let bitindex = 0; bitindex < 10; bitindex++){
				let bx = (mortoncode >> (2 * bitindex + 0)) & 1;
				let by = (mortoncode >> (2 * bitindex + 1)) & 1;

				mx = mx | (bx << bitindex);
				my = my | (by << bitindex);
			}

			// let decode = (value) => {

			// 	let sign = 1;
			// 	if(value > 7){
			// 		sign = -1;
			// 		value = value - 8;
			// 	}

			// 	value = sign * (2 ** value);

			// 	return value;
			// };

			// debugger;

			// view_target.setUint16(offset_rgb + 6 * i + 0, decode(view_colBuffer.getUint8(3 * i + 0)), true);
			// view_target.setUint16(offset_rgb + 6 * i + 2, decode(view_colBuffer.getUint8(3 * i + 1)), true);
			// view_target.setUint16(offset_rgb + 6 * i + 4, decode(view_colBuffer.getUint8(3 * i + 2)), true);
			// 376968
			// debugger;
			view_target.setUint16(offset_rgb + 6 * i + 0, view_colBuffer.getUint8(3 * i + 0), true);
			view_target.setUint16(offset_rgb + 6 * i + 2, view_colBuffer.getUint8(3 * i + 1), true);
			view_target.setUint16(offset_rgb + 6 * i + 4, view_colBuffer.getUint8(3 * i + 2), true);


			// let color = readPixel(mx, my);

			// view_target.setUint16(offset_rgb + 6 * i + 0, color[0], true);
			// view_target.setUint16(offset_rgb + 6 * i + 2, color[1], true);
			// view_target.setUint16(offset_rgb + 6 * i + 4, color[2], true);
		}

	}else{
		// child voxels encoded relative to parent voxels

		let nodeIndex = Number(node.name.slice(-1));

		// let parentVoxels = {
		// 	0: [], 1: [], 2: [], 3: [],
		// 	4: [], 5: [], 6: [], 7: [],
		// };

		let view_parent = new DataView(parent.buffer);
		let parentSize = getBoxSize(parent.min, parent.max);

		// let prevChildIndex = 0;

		// for(let i = 0; i < parent.numVoxels; i++){
		// 	let x = view_parent.getFloat32(12 * i + 0, true);
		// 	let y = view_parent.getFloat32(12 * i + 4, true);
		// 	let z = view_parent.getFloat32(12 * i + 8, true);

		// 	// PROTOTYPING
		// 	// debugger;
		// 	// if(offset_rgb + 6 * i + 4 + 2 >= view_parent.byteLength) debugger;
		// 	let r = view_parent.getUint16(12 * parent.numVoxels + 6 * i + 0, true);
		// 	let g = view_parent.getUint16(12 * parent.numVoxels + 6 * i + 2, true);
		// 	let b = view_parent.getUint16(12 * parent.numVoxels + 6 * i + 4, true);

		// 	let vx = 2 * (x - parent.min[0]) / parentSize[0];
		// 	let vy = 2 * (y - parent.min[1]) / parentSize[1];
		// 	let vz = 2 * (z - parent.min[2]) / parentSize[2];

		// 	vx = Math.min(Math.floor(vx), 1);
		// 	vy = Math.min(Math.floor(vy), 1);
		// 	vz = Math.min(Math.floor(vz), 1);

		// 	let childIndex = (vx << 2) | (vy << 1) | (vz << 0);

		// 	if(childIndex < 0 || childIndex > 7){
		// 		debugger;
		// 	}

		// 	// debugger;
		// 	// console.log(childIndex);
		// 	if(childIndex !== prevChildIndex){
		// 		console.log(childIndex);
		// 		prevChildIndex = childIndex;
		// 	}

		// 	let voxel = {x, y, z, r, g, b, i};
		// 	// debugger;

		// 	// if(childIndex === nodeIndex){
		// 		parentVoxels[childIndex].push(voxel);
		// 	// }
		// }

		// debugger;

		let prefixsum = [0];
		for(let i = 0; i < 8; i++){
			prefixsum[i + 1] = prefixsum[i] + parent.numVoxelsPerOctant[i];
		}

		let childVoxels = [];

		// if(parent.numVoxelsPerOctant[nodeIndex] !== parentVoxels[nodeIndex].length){
		// 	debugger;
		// }

		// if(node.name === "r00"){
		// 	debugger;
		// }
		
		for(let i = 0; i < parent.numVoxelsPerOctant[nodeIndex]; i++){
			// let parentVoxel = parentVoxels[nodeIndex][i];
			// debugger;

			// if(childVoxels.length >= node.numVoxels) break;

			let poffset = prefixsum[nodeIndex];

			let parent_x = view_parent.getFloat32(12 * (i + poffset) + 0, true);
			let parent_y = view_parent.getFloat32(12 * (i + poffset) + 4, true);
			let parent_z = view_parent.getFloat32(12 * (i + poffset) + 8, true);

			// if(parent_x !== parentVoxel.x){
			// 	// debugger;
			// 	console.error("parent_x !== parentVoxel.x");
			// 	return;
			// }

			let childmask = view_voxel.getUint8(i);

			for(let childIndex = 0; childIndex < 8; childIndex++){
				let bit = (childmask >> childIndex) & 1;

				if(bit === 1){
					let bx = (childIndex >> 2) & 1;
					let by = (childIndex >> 1) & 1;
					let bz = (childIndex >> 0) & 1;

					let childCoordOffset = {x: 0, y: 0, z: 0};
					if(bx == 0){
						childCoordOffset.x = -1;
					}else{
						childCoordOffset.x =  1;
					}
					if(by == 0){
						childCoordOffset.y = -1;
					}else{
						childCoordOffset.y =  1;
					}
					if(bz == 0){
						childCoordOffset.z = -1;
					}else{
						childCoordOffset.z =  1;
					}
					
					childCoordOffset.x = childCoordOffset.x * nodeSpacing * 0.5;
					childCoordOffset.y = childCoordOffset.y * nodeSpacing * 0.5;
					childCoordOffset.z = childCoordOffset.z * nodeSpacing * 0.5;

					// debugger;
					let childCoord = {
						x: (parent_x + childCoordOffset.x),
						y: (parent_y + childCoordOffset.y),
						z: (parent_z + childCoordOffset.z),
					};

					// PROTOTYPING
					let childVoxel = {
						x: childCoord.x, y: childCoord.y, z: childCoord.z,
						// parent: parentVoxel
					};

					childVoxels.push(childVoxel);
				}

			}

		}

		// let clamp = (value, min, max) => {
		// 	if(value < min) return min;
		// 	if(value > max) return max;
			
		// 	return value;
		// };

		// let parentSize = getBoxSize(parent.min, parent.max);
		// let toVoxelIndex = (x, y, z) => {

		// 	let ix = Math.floor(clamp(128 * (x - parent.min.x) / (parentSize.x), 0, 127));
		// 	let iy = Math.floor(clamp(128 * (y - parent.min.y) / (parentSize.y), 0, 127));
		// 	let iz = Math.floor(clamp(128 * (z - parent.min.z) / (parentSize.z), 0, 127));
			
		// 	let voxelIndex = ix + iy * 128 + iz * 128 * 128;
		
		// 	return voxelIndex;
		// };

		// let i_parent = 0;
		// for(let i_child = 0; i_child < node.numVoxels; i_child++){

		// 	let parent_x = view_parent.getFloat32(12 * i + 0, true);
		// 	let parent_y = view_parent.getFloat32(12 * i + 4, true);
		// 	let parent_z = view_parent.getFloat32(12 * i + 8, true);

		// 	let isParent = toVoxelIndex(v_parent) == toVoxelIndex(v_child);

		// }


		if(childVoxels.length !== node.numVoxels){
			console.assert(`reproduced wrong amount of voxels. expected: ${node.numVoxels}, got: ${childVoxels.length}`);
			debugger;
		}

		for(let i = 0; i < childVoxels.length; i++){
			let childVoxel = childVoxels[i];

			if(childVoxel.x == 0 && childVoxel.y == 0 && childVoxel.z == 0){
				debugger;
			}

			view_target.setFloat32(offset_xyz + 12 * i + 0, childVoxel.x, true);
			view_target.setFloat32(offset_xyz + 12 * i + 4, childVoxel.y, true);
			view_target.setFloat32(offset_xyz + 12 * i + 8, childVoxel.z, true);

			let mortoncode = i;
				
			let x = 0;
			let y = 0;
			for(let bitindex = 0; bitindex < 10; bitindex++){
				let bx = (mortoncode >> (2 * bitindex + 0)) & 1;
				let by = (mortoncode >> (2 * bitindex + 1)) & 1;

				x = x | (bx << bitindex);
				y = y | (by << bitindex);
			}


			// let decode = (value) => {

			// 	let sign = 1;
			// 	if(value > 7){
			// 		sign = -1;
			// 		value = value - 8;
			// 	}

			// 	value = sign * (2 ** value);

			// 	return value;
			// };

			// if(node.name === "r0"){
			// 	debugger;
			// }

			// view_target.setUint16(offset_rgb + 6 * i + 0, childVoxel.parent.r + decode(view_colBuffer.getUint8(3 * i + 0)));
			// view_target.setUint16(offset_rgb + 6 * i + 2, childVoxel.parent.g + decode(view_colBuffer.getUint8(3 * i + 1)));
			// view_target.setUint16(offset_rgb + 6 * i + 4, childVoxel.parent.b + decode(view_colBuffer.getUint8(3 * i + 2)));

			view_target.setUint16(offset_rgb + 6 * i + 0, view_colBuffer.getUint8(3 * i + 0));
			view_target.setUint16(offset_rgb + 6 * i + 2, view_colBuffer.getUint8(3 * i + 1));
			view_target.setUint16(offset_rgb + 6 * i + 4, view_colBuffer.getUint8(3 * i + 2));

			// let color = readPixel(x, y);

			// { // PROTOTYPING: LOGRATHMIC ENCODING

			// 	// first difference-encoding to parent voxel
			// 	let diffR = color[0] - childVoxel.parent.r;
			// 	let diffG = color[1] - childVoxel.parent.g;
			// 	let diffB = color[2] - childVoxel.parent.b;

			// 	// then take log2 of difference and quantize to integer
			// 	let {abs, log2, sign, round, pow} = Math;
			// 	let diffR_i = round(log2(abs(diffR)));
			// 	let diffG_i = round(log2(abs(diffG)));
			// 	let diffB_i = round(log2(abs(diffB)));

			// 	// see what happens when we reconstruct the color from the quantized, log2 and difference encoded color
			// 	let recoveredR = sign(diffR) * pow(2, diffR_i) + childVoxel.parent.r;
			// 	let recoveredG = sign(diffG) * pow(2, diffG_i) + childVoxel.parent.g;
			// 	let recoveredB = sign(diffB) * pow(2, diffB_i) + childVoxel.parent.b;

			// 	view_target.setUint16(offset_rgb + 6 * i + 0, recoveredR, true);
			// 	view_target.setUint16(offset_rgb + 6 * i + 2, recoveredG, true);
			// 	view_target.setUint16(offset_rgb + 6 * i + 4, recoveredB, true);
			// }

			// { // PROTOTYPING
			// 	let diffR = color[0] - childVoxel.parent.r;
			// 	let diffG = color[1] - childVoxel.parent.g;
			// 	let diffB = color[2] - childVoxel.parent.b;

			// 	let {abs, log2, sign, round, pow} = Math;
			// 	// let diffR_i = round(log2(abs(diffR)));
			// 	// let diffG_i = round(log2(abs(diffG)));
			// 	// let diffB_i = round(log2(abs(diffB)));

			// 	let recoveredR = diffR + childVoxel.parent.r;
			// 	let recoveredG = diffG + childVoxel.parent.g;
			// 	let recoveredB = diffB + childVoxel.parent.b;

			// 	view_target.setUint16(offset_rgb + 6 * i + 0, recoveredR, true);
			// 	view_target.setUint16(offset_rgb + 6 * i + 2, recoveredG, true);
			// 	view_target.setUint16(offset_rgb + 6 * i + 4, recoveredB, true);
			// }

			// NORMAL ENCODING
			// view_target.setUint16(offset_rgb + 6 * i + 0, color[0], true);
			// view_target.setUint16(offset_rgb + 6 * i + 2, color[1], true);
			// view_target.setUint16(offset_rgb + 6 * i + 4, color[2], true);
		}

	}

	let tEnd = performance.now();
	let ms = tEnd - tStart;
	// let msJpeg = dJpeg;
	let msJpeg         = 0;
	let msVoxels       = tEnd - tVoxelStart;
	let mVoxelsSec     = (1000 * node.numVoxels / ms) / 1_000_000;
	let strName        = node.name.padEnd(5, " ");
	let strVoxels      = node.numVoxels.toLocaleString().padStart(8, " ");
	// let strColor   = msJpeg.toFixed(1);
	// let strColor       = "0";
	// let strColor       = dDecompressColor.toFixed(1).padStart(4, " ");
	let strColor       = "0";
	let strDVoxels     = msVoxels.toFixed(1).padStart(5, " ");
	let strDTotal      = ms.toFixed(1).padStart(5, " ");
	let strThroughput  = mVoxelsSec.toFixed(2).padStart(6);
	console.log(`node: ${strName}, #voxels: ${strVoxels}, dColor: ${strColor}ms, dVoxels: ${strDVoxels}ms, dTotal: ${strDTotal}ms, voxels/sec: ${strThroughput} M`);


	node.buffer = target.slice();
	// console.log("loaded ", node.name);

	let message = {
		type: "node parsed",
		node, buffer: target
	};
	let transferables = [target];

	postMessage(message, transferables);
}

async function load(event){

	let response = await fetch(event.data.url);
	let buffer = await response.arrayBuffer();

	let nodeMap = new Map();
	if(event.data.parent){
		nodeMap.set(event.data.parent.name, event.data.parent);
	}

	for(let node of event.data.nodes){

		let parentName = node.name.slice(0, node.name.length - 1);
		let parent = nodeMap.get(parentName) ?? null;
		let nodeLevel = node.name.length - 1;
		let nodeSpacing = event.data.spacing / (2 ** nodeLevel);

		await loadNode(node, nodeSpacing, parent, buffer);

		nodeMap.set(node.name, node);
	}

	postMessage({type: "batch finished"});


	return event.data.nodes;
}

onmessage = async function (event) {

	try{
		let nodes = await load(event);

		let message = {
			nodes,
		};

	}catch(e){
		console.log(e);
		postMessage("failed");
	}

	
};
