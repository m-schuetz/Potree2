
// import {Box3, Vector3} from "potree";

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

	let view_voxel = new DataView(buffer, node.voxelBufferOffset);
	let view_jpeg = new DataView(buffer, node.jpegBufferOffset);

	let boxSize = getBoxSize(node.min, node.max);

	let target = new ArrayBuffer(node.numVoxels * 20);
	let view_target = new DataView(target);

	let offset_xyz = 0;
	let offset_rgb = 12 * node.numVoxels;

	let imageData;
	{ // try loading image
		let blob = new Blob([new Uint8Array(buffer, node.jpegBufferOffset, node.jpegBufferSize)]);
		let bitmap = await createImageBitmap(blob);

		let [width, height] = [bitmap.width, bitmap.height];

		const canvas = new OffscreenCanvas(256, 256);
		const context = canvas.getContext("2d");
		context.drawImage(bitmap, 0, 0);

		imageData = context.getImageData(0, 0, width, height);
	}

	let readPixel = (x, y) => {
		let pixelID = x + imageData.width * y;

		let r = imageData.data[4 * pixelID + 0];
		let g = imageData.data[4 * pixelID + 1];
		let b = imageData.data[4 * pixelID + 2];
		let a = imageData.data[4 * pixelID + 3];

		return [r, g, b, a];
	};

	if(node.name === "r66440") debugger;

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

			let color = readPixel(mx, my);

			view_target.setUint16(offset_rgb + 6 * i + 0, color[0], true);
			view_target.setUint16(offset_rgb + 6 * i + 2, color[1], true);
			view_target.setUint16(offset_rgb + 6 * i + 4, color[2], true);
		}

	}else{
		// child voxels encoded relative to parent voxels

		let parentVoxels = {
			0: [], 1: [], 2: [], 3: [],
			4: [], 5: [], 6: [], 7: [],
		};

		let view_parent = new DataView(parent.buffer);
		let parentSize = getBoxSize(parent.min, parent.max);

		for(let i = 0; i < parent.numVoxels; i++){
			let x = view_parent.getFloat32(12 * i + 0, true);
			let y = view_parent.getFloat32(12 * i + 4, true);
			let z = view_parent.getFloat32(12 * i + 8, true);

			let vx = 2 * (x - parent.min[0]) / parentSize[0];
			let vy = 2 * (y - parent.min[1]) / parentSize[1];
			let vz = 2 * (z - parent.min[2]) / parentSize[2];

			vx = Math.min(Math.floor(vx), 1);
			vy = Math.min(Math.floor(vy), 1);
			vz = Math.min(Math.floor(vz), 1);

			let childIndex = (vx << 2) | (vy << 1) | (vz << 0);

			let voxel = {x, y, z};
			parentVoxels[childIndex].push(voxel);
		}

		let nodeIndex = Number(node.name.slice(-1));
		let childVoxels = [];
		for(let i = 0; i < parentVoxels[nodeIndex].length; i++){
			let parentVoxel = parentVoxels[nodeIndex][i];
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

					let childCoord = {
						x: (parentVoxel.x + childCoordOffset.x),
						y: (parentVoxel.y + childCoordOffset.y),
						z: (parentVoxel.z + childCoordOffset.z),
					};

					childVoxels.push(childCoord);
				}

			}
		}

		for(let i = 0; i < childVoxels.length; i++){
			let pos = childVoxels[i];

			view_target.setFloat32(offset_xyz + 12 * i + 0, pos.x, true);
			view_target.setFloat32(offset_xyz + 12 * i + 4, pos.y, true);
			view_target.setFloat32(offset_xyz + 12 * i + 8, pos.z, true);

			let mortoncode = i;
				
			let x = 0;
			let y = 0;
			for(let bitindex = 0; bitindex < 10; bitindex++){
				let bx = (mortoncode >> (2 * bitindex + 0)) & 1;
				let by = (mortoncode >> (2 * bitindex + 1)) & 1;

				x = x | (bx << bitindex);
				y = y | (by << bitindex);
			}

			let color = readPixel(x, y);

			view_target.setUint16(offset_rgb + 6 * i + 0, color[0], true);
			view_target.setUint16(offset_rgb + 6 * i + 2, color[1], true);
			view_target.setUint16(offset_rgb + 6 * i + 4, color[2], true);
		}

	}

	node.buffer = target.slice();
	// console.log("loaded ", node.name);

	let message = {
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

	// return {
	// 	buffer: new Uint8Array(outBuffer), statsList
	// };


	return event.data.nodes;
}

onmessage = async function (event) {

	try{
		let nodes = await load(event);

		let message = {
			nodes,
		};
		
		// let transferables = [];

		// for(let node of nodes){
		// 	transferables.push(node.buffer);
		// }

		// transferables.push(loaded.buffer.buffer);

		// postMessage(message, transferables);
	}catch(e){
		console.log(e);
		postMessage("failed");
	}

	
};
