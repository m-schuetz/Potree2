
import {PointCloudOctree} from "potree";
import {PointCloudOctreeNode} from "./PointCloudOctreeNode.js";
import {PointAttribute, PointAttributes, PointAttributeTypes} from "./PointAttributes.js";
import {WorkerPool} from "../misc/WorkerPool.js";
import {Geometry} from "potree";
import {Vector3, Box3, Matrix4} from "potree";
import JSON5 from "json5";

const NodeType = {
	NORMAL: 0,
	LEAF: 1,
	PROXY: 2,
};

function round4(number){
	return number + (4 - (number % 4));
}

let SPECTRAL = [
	[213,62,79],
	[252,141,89],
	[254,224,139],
	[230,245,152],
	[153,213,148],
	[50,136,189],
];

export class Potree2Loader{

	constructor(){
		this.metadata = null;
	}

	loadNode(node){

	}

	static async load(url){
		let loader = new Potree2Loader();
		loader.url = url;

		let response = await fetch(url + "/metadata.json");
		// let metadata = await response.json();
		let text = await response.text();

		let metadata = JSON5.parse(text);


		let attributes;
		{
			let a_xyz = new PointAttribute("position", PointAttributeTypes.FLOAT, 3);
			a_xyz.range = [
				metadata.boundingBox.min,
				metadata.boundingBox.max,
			];

			let a_rgb = new PointAttribute("rgba", PointAttributeTypes.UINT16, 3);
			a_rgb.range = [
				[0, 0, 0],
				[255, 255, 255],
			];

			attributes = new PointAttributes([a_xyz, a_rgb]);
		}

		loader.metadata = metadata;
		loader.attributes = attributes;
		// loader.scale = metadata.scale;
		// loader.offset = metadata.offset;

		let octree = new PointCloudOctree();
		octree.url = url;
		octree.spacing = metadata.spacing;
		octree.boundingBox = new Box3(
			new Vector3(...metadata.boundingBox.min),
			new Vector3(...metadata.boundingBox.max),
		);
		octree.position.copy(octree.boundingBox.min);
		octree.boundingBox.max.sub(octree.boundingBox.min);
		octree.boundingBox.min.set(0, 0, 0);
		octree.updateWorld();

		octree.attributes = attributes;
		octree.loader = loader;
		loader.octree = octree;
		octree.material.init(octree);

		// let root = new PointCloudOctreeNode("r");
		// root.boundingBox = octree.boundingBox.clone();
		// root.level = 0;
		// root.nodeType = NodeType.NORMAL;
		// root.spacing = octree.spacing;
		// root.byteOffset = 0;
		// root.octree = octree;

		// sort by level

		// LOAD NODES
		let nodeMap = new Map();
		let metanodeMap = new Map();
		for(let metanode of metadata.nodes){

			// if(metanode.name.length > 2) continue;

			let node = new PointCloudOctreeNode(metanode.name);
			node.level = node.name.length - 1;
			node.boundingBox = new Box3(
				new Vector3(...metanode.min),
				new Vector3(...metanode.max),
			);
			node.nodeType = NodeType.NORMAL;
			node.spacing = metadata.spacing / (2.0 ** node.level);
			node.octree = octree;
			node.loaded = false;

			nodeMap.set(node.name, node);
			metanodeMap.set(node.name, metanode);
		}

		// connect nodes
		for(let [nodename, node] of nodeMap){

			if(nodename === "r") continue;

			let parentName = nodename.substr(0, nodename.length - 1);
			let parent = nodeMap.get(parentName);

			let childIndex = Number(nodename.slice(-1));
			parent.children[childIndex] = node;
			node.parent = parent;
		}

		let root = nodeMap.get("r");

		let nodes = Array.from(nodeMap.values());
		nodes.sort((a, b) => {
			return a.level - b.level;
		});
		
		// load geometry
		// root.traverse(async node => {
		for(let node of nodes){

			let metanode = metanodeMap.get(node.name);

			console.log(`processing ${node.name}, #voxels: ${metanode.numVoxels}`);

			if(node.name === "r"){
				let n = metanode.numVoxels;
				let bufferSize = round4(18 * n);
				let buffer = new Uint8Array(bufferSize);

				let response = await fetch(`${url}/${node.name}.voxels`);
				let source = await response.arrayBuffer();
				let sourceView = new DataView(source);
				let targetView = new DataView(buffer.buffer);
				let offset_xyz = 0;
				let offset_rgb = 12 * n;

				for(let i = 0; i < n; i++){
					let x = sourceView.getFloat32(12 * i + 0, true);
					let y = sourceView.getFloat32(12 * i + 4, true);
					let z = sourceView.getFloat32(12 * i + 8, true);
					let r = 255;
					let g = 0;
					let b = 0;

					targetView.setFloat32(offset_xyz + 12 * i + 0, x, true);
					targetView.setFloat32(offset_xyz + 12 * i + 4, y, true);
					targetView.setFloat32(offset_xyz + 12 * i + 8, z, true);
					targetView.setUint16(offset_rgb + 6 * i + 0, r, true);
					targetView.setUint16(offset_rgb + 6 * i + 2, g, true);
					targetView.setUint16(offset_rgb + 6 * i + 4, b, true);
				}

				let geometry = new Geometry();
				geometry.numElements = n;
				geometry.buffer = buffer;
				node.geometry = geometry;

			}else if(metanode.numVoxels > 0){

				// let groundtruth = [];
				// {
				// 	let n = metanode.numVoxels;
				// 	// let bufferSize = round4(18 * n);
				// 	// let buffer = new Uint8Array(bufferSize);

				// 	let response = await fetch(`${url}/${node.name}.full.voxels`);
				// 	let source = await response.arrayBuffer();
				// 	let sourceView = new DataView(source);
				// 	// let targetView = new DataView(buffer.buffer);
				// 	// let offset_xyz = 0;
				// 	// let offset_rgb = 12 * n;

				// 	for(let i = 0; i < n; i++){
				// 		let x = sourceView.getFloat32(12 * i + 0, true);
				// 		let y = sourceView.getFloat32(12 * i + 4, true);
				// 		let z = sourceView.getFloat32(12 * i + 8, true);

				// 		groundtruth.push(new Vector3(x, y, z));
				// 		// let [r, g, b] = SPECTRAL[node.level];

				// 		// targetView.setFloat32(offset_xyz + 12 * i + 0, x, true);
				// 		// targetView.setFloat32(offset_xyz + 12 * i + 4, y, true);
				// 		// targetView.setFloat32(offset_xyz + 12 * i + 8, z, true);
				// 		// targetView.setUint16(offset_rgb + 6 * i + 0, r, true);
				// 		// targetView.setUint16(offset_rgb + 6 * i + 2, g, true);
				// 		// targetView.setUint16(offset_rgb + 6 * i + 4, b, true);

				// 		// if(i === 0 && node.name === "r0"){
				// 		// 	console.log(`r0.voxels[0]: ${x}, ${y}, ${z}`);
				// 		// }
				// 	}

				// 	// let geometry = new Geometry();
				// 	// geometry.numElements = n;
				// 	// geometry.buffer = buffer;
				// 	// node.geometry = geometry;
				// }



				// each byte is a childmask. Number of 1-bits = number of voxels
				// 1. find morton-sorted voxel coordinates in parent
				// 2. Each voxel in parent splits into 8 child voxels. The childmask specifies which ones.


				let response = await fetch(`${url}/${node.name}.voxels`);
				let source = await response.arrayBuffer();
				let sourceView = new DataView(source);


				let parentVoxels = {
					0: [], 1: [], 2: [], 3: [],
					4: [], 5: [], 6: [], 7: [],
				};
				let parent = node.parent;
				let parentMetanode = metanodeMap.get(parent.name);
				let parentView = new DataView(parent.geometry.buffer.buffer);
				let box = node.boundingBox;
				let parentSize = parent.boundingBox.size();
				let gridSize = 128;
				let nodeIndex = Number(node.name.slice(-1));

				for(let i = 0; i < parentMetanode.numVoxels; i++){
					let x = parentView.getFloat32(12 * i + 0, true);
					let y = parentView.getFloat32(12 * i + 4, true);
					let z = parentView.getFloat32(12 * i + 8, true);

					let vx = 2 * (x - parent.boundingBox.min.x) / parentSize.x;
					let vy = 2 * (y - parent.boundingBox.min.y) / parentSize.y;
					let vz = 2 * (z - parent.boundingBox.min.z) / parentSize.z;
					vx = Math.min(Math.floor(vx), 1);
					vy = Math.min(Math.floor(vy), 1);
					vz = Math.min(Math.floor(vz), 1);

					let childIndex = (vx << 2) | (vy << 1) | (vz << 0);

					let voxel = new Vector3(x, y, z);
					parentVoxels[childIndex].push(voxel);

				}


				let childVoxels = [];
				// FIXME: source.byteLength should be same as parentVoxels.length but isnt!!!
				for(let i = 0; i < source.byteLength; i++)
				// for(let i = 0; i < parentVoxels.length; i++)
				{
					let parentVoxel = parentVoxels[nodeIndex][i];
					let childmask = sourceView.getUint8(i);

					for(let childIndex = 0; childIndex < 8; childIndex++){
						let bit = (childmask >> childIndex) & 1;

						if(bit === 1){
							let bx = (childIndex >> 2) & 1;
							let by = (childIndex >> 1) & 1;
							let bz = (childIndex >> 0) & 1;

							let childCoordOffset = new Vector3();
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
							
							childCoordOffset.multiplyScalar(node.spacing * 0.5);
							let childCoord = parentVoxel.clone().add(childCoordOffset);

							// if(childCoord.distanceTo(groundtruth[childVoxels.length]) > 0.001){
							// 	debugger;
							// }

							childVoxels.push(childCoord);
						}

					}

				}

				let n = childVoxels.length;
				let bufferSize = round4(18 * n);
				let buffer = new Uint8Array(bufferSize);
				let view = new DataView(buffer.buffer);
				let boxSize = node.boundingBox.size();
				let offset_xyz = 0;
				let offset_rgb = 12 * n;

				for(let i = 0; i < n; i++){
					let pos = childVoxels[i];
					let [r, g, b] = SPECTRAL[node.level];

					view.setFloat32(offset_xyz + 12 * i + 0, pos.x, true);
					view.setFloat32(offset_xyz + 12 * i + 4, pos.y, true);
					view.setFloat32(offset_xyz + 12 * i + 8, pos.z, true);
					view.setUint16(offset_rgb + 6 * i + 0, r, true);
					view.setUint16(offset_rgb + 6 * i + 2, g, true);
					view.setUint16(offset_rgb + 6 * i + 4, b, true);
				}

				let geometry = new Geometry();
				geometry.numElements = n;
				geometry.buffer = buffer;
				node.geometry = geometry;

			}else{

				let n = 1000;
				let bufferSize = round4(18 * n);
				let buffer = new Uint8Array(bufferSize);
				let view = new DataView(buffer.buffer);
				let boxSize = node.boundingBox.size();
				let offset_xyz = 0;
				let offset_rgb = 12 * n;
				for(let i = 0; i < n; i++){
					let x = Math.random() * boxSize.x + node.boundingBox.min.x;
					let y = Math.random() * boxSize.y + node.boundingBox.min.y;
					let z = Math.random() * boxSize.z + node.boundingBox.min.z;
					z = node.boundingBox.min.z;
					let r = 255;
					let g = 0;
					let b = 0;

					view.setFloat32(offset_xyz + 12 * i + 0, x, true);
					view.setFloat32(offset_xyz + 12 * i + 4, y, true);
					view.setFloat32(offset_xyz + 12 * i + 8, z, true);
					view.setUint16(offset_rgb + 6 * i + 0, r, true);
					view.setUint16(offset_rgb + 6 * i + 2, g, true);
					view.setUint16(offset_rgb + 6 * i + 4, b, true);
				}

				let geometry = new Geometry();
				geometry.numElements = n;
				geometry.buffer = buffer;
				node.geometry = geometry;
			}

			node.loaded = true;
		}

		octree.root = root;

		Potree.events.dispatcher.dispatch("pointcloud_loaded", octree);

		return octree;
	}

}