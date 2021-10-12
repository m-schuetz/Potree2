
import {Box3, Vector3, Mesh, PhongMaterial} from "potree";
import {renderVoxelsLOD} from "./renderVoxelsLOD.js";
import {renderVoxelsLOD_quads} from "./renderVoxelsLOD_quads.js";
import {renderVoxelsLOD_points} from "./renderVoxelsLOD_points.js";
import {BinaryHeap} from "BinaryHeap";

// let metadataFilename = "metadata_sitn.json";
// let dataFilename = "data_sitn.bin";

let metadataFilename = "metadata.json";
let dataFilename = "data.bin";

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

class Node{

	constructor(){
		this.name = "";
		this.boundingBox = new Box3();
		this.voxels = null;
		this.mesh = null;
		this.parent = null;
		this.children = new Array(8).fill(null);
		this.isLeaf = true;
		this.level = 0;
		this.voxelGridSize = 0;
		this.visible = false;
		this.loading = false;
		this.loaded = false;
	}

	traverse(callback){
		callback(this);

		for(let child of this.children){
			if(child){
				child.traverse(callback);
			}
		}
	}

}

let numBytesLoaded = 0;


export class Potree3Loader{

	constructor(){

	}

	static async load(url){

		let rMetadata = await fetch(`${url}/${metadataFilename}`);
		let jsMetadata = await rMetadata.json();

		let root = null;
		let nodes = [];
		let nodeMap = new Map();

		for(let jsNode of jsMetadata.nodes){
			let node = new Node();
			node.name = jsNode.name;
			node.level = node.name.length - 1;
			node.byteOffset = jsNode.offset;
			node.byteSize = jsNode.size;

			nodes.push(node);
			nodeMap.set(jsNode.name, node);
		}

		for(let node of nodes){
			
			if(node.name === "r"){
				root = node;
			}else{
				let parentName = node.name.slice(0, -1);
				let parentNode = nodeMap.get(parentName);

				let childIndex = Number(node.name.slice(-1, Infinity));

				parentNode.children[childIndex] = node;
				node.parent = parentNode;
			}
		}

		root.boundingBox = new Box3(
			new Vector3(...jsMetadata.boundingBox.min),
			new Vector3(...jsMetadata.boundingBox.max),
		);

		let rootVoxelSize = root.boundingBox.size().x / jsMetadata.gridSize;

		root.traverse( (node) => {
			
			for(let childIndex = 0; childIndex < 8; childIndex++){
				let child = node.children[childIndex];

				if(child){
					node.isLeaf = false;

					child.boundingBox = createChildAABB(node.boundingBox, childIndex);
				}
			}

			node.voxelGridSize = jsMetadata.gridSize;

		});

		root.traverse( (node) => {

			node.load = async () => {
				// load voxels

				if(node.loading){
					return;
				}else{
					node.loading = true;
				}

				try{
					let first = node.byteOffset;
					let last = first + node.byteSize - 1;

					let response = await fetch(`${url}/${dataFilename}`, {
						headers: {
							'content-type': 'multipart/byteranges',
							'Range': `bytes=${first}-${last}`,
						},
					});

					let numVoxels = node.byteSize / 16;
					let buffer = await response.arrayBuffer();

					let positions = new Float32Array(buffer, 0, 3 * numVoxels);
					let colors = new Uint8Array(buffer, 12 * numVoxels, 4 * numVoxels);

					node.voxels = {positions, colors, numVoxels};
					node.loaded = true;

					node.load = () => {};
					node.loading = false;
					numBytesLoaded += node.byteSize;

					console.log(`[${node.name}]: ${numVoxels} voxels; ${Math.floor(node.byteSize / 1024)} kb`);
					console.log(`mb loaded: ${numBytesLoaded / (1024 * 1024)}`);
				}catch(e){
					console.log(`failed to load ${node.name}. Trying again.`);
					console.log(e);
					node.loading = false;
				}
			}
				
		});

		let visibleNodes = [];

		potree.onUpdate( () => {

			if(!guiContent["update"]){
				return;
			}

			let numVisibleNodes = 0;
			let numVisibleVoxels = 0;


			root.traverse( (node) => { 
				node.visible = false;
			} );


			let priorityQueue = new BinaryHeap((element) => {element.weight});
			priorityQueue.push({node: root, weight: 1});

			visibleNodes = [];
			while (priorityQueue.size() > 0) {

				let element = priorityQueue.pop();
				let node = element.node;


				if(!node.loaded){
					node.load();
					node.visible = false;

					continue;
				}

				node.visible = true;
				numVisibleNodes++;
				if(node.voxels){
					numVisibleVoxels += node.voxels.numVoxels;
					visibleNodes.push(node);
				}

				for(let child of node.children){

					if (!child) continue;

					let center = child.boundingBox.center();
					let size = child.boundingBox.size().length();
					let camWorldPos = camera.getWorldPosition();
					let distance = camWorldPos.distanceTo(center);
					let weight = (size / distance);
					let visible = weight > 0.3 / Potree.settings.debugU;

					// visible = ["r", "r0", "r00", "r006", "r0060", "r0061"].includes(child.name);

					if(visible){
						priorityQueue.push({node: child, weight: weight});
					}

				}


			}

			Potree.state.numVoxels = numVisibleVoxels;
			// guiContent["#points"] = numVisibleVoxels.toLocaleString();
			guiContent["#nodes"] = numVisibleNodes.toLocaleString();

			
		});

		potree.onUpdate(() => {
			let showBoundingBox = guiContent["show bounding box"];

			if(showBoundingBox){

				for(let node of visibleNodes){
					potree.renderer.drawBoundingBox(
						node.boundingBox.center(),
						node.boundingBox.size(),
						new Vector3(255, 255, 0),
					);
				}
			}
		});

		potree.renderer.onDraw(drawstate => {
			renderVoxelsLOD(root, drawstate);
			// renderVoxelsLOD_points(root, drawstate);
			// renderVoxelsLOD_quads(root, drawstate);
		});

	}

}