
import {Box3, Vector3, Matrix4, Frustum} from "potree";
import {renderVoxelsLOD} from "./renderVoxelsLOD.js";
import {renderVoxelsLOD_quads} from "./renderVoxelsLOD_quads.js";
import {renderVoxelsLOD_points} from "./renderVoxelsLOD_points.js";
import {BinaryHeap} from "BinaryHeap";
import {BrotliDecode} from "../../../libs/brotli/decode.js";
import {WorkerPool} from "../../misc/WorkerPool.js";

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

let numBatchesLoading = 0;
let maxNumBatchesLoading = 4;

class Batch{

	constructor(){
		this.boundingBox = new Box3();
		this.name = "";
		this.children = [];
		this.nodes = [];

		this.url = "";
		this.posOffset = 0;
		this.posSize = 0;
		this.colOffset = 0;
		this.colSize = 0;
		this.loaded = false;
		this.loading = false;

	}

	async load(){

		if(this.loaded) return;
		if(this.loading) return;
		if(numBatchesLoading > maxNumBatchesLoading) return;

		this.loading = true;
		numBatchesLoading++;


		let workerPath = "./src/misc/potree3loader_batched/Potree3LoaderWorker.js";
		let worker = WorkerPool.getWorker(workerPath, {type: "module"});

		worker.onmessage = (e) => {
			let data = e.data;

			let position_decoded = data.position_decoded;
			let color_decoded = data.color_decoded;

			let position_decoded_u32 = new Int32Array(position_decoded.buffer);

			let numVoxels_batch = position_decoded.byteLength / 4;
			let positions = new Float32Array(3 * numVoxels_batch);
			let voxelsProcessed = 0;

			for(let node of this.nodes){
				let box = node.boundingBox;
				let cubeSize = box.max.x - box.min.x;
				let voxelGridSize = node.voxelGridSize;

				for(let i = 0; i < node.numVoxels; i++){

					let voxelIndex = position_decoded_u32[voxelsProcessed];
					let X = voxelIndex % voxelGridSize;
					let Y = Math.floor((voxelIndex % (voxelGridSize * voxelGridSize)) / voxelGridSize);
					let Z = Math.floor(voxelIndex / (voxelGridSize * voxelGridSize));

					let x = cubeSize * (X / voxelGridSize) + box.min.x;
					let y = cubeSize * (Y / voxelGridSize) + box.min.y;
					let z = cubeSize * (Z / voxelGridSize) + box.min.z;

					positions[3 * voxelsProcessed + 0] = x;
					positions[3 * voxelsProcessed + 1] = y;
					positions[3 * voxelsProcessed + 2] = z;

					voxelsProcessed++;
				}

			}
			
			this.buffers = {
				"position": positions,
				"color": color_decoded,
			};

			this.loaded = true;
			this.loading = false;
			numBatchesLoading--;
			WorkerPool.returnWorker(workerPath, worker);
		};

		let url = new URL(this.url, document.baseURI).href;
		let message = {
			url: url,
			posOffset: this.posOffset,
			posSize: this.posSize,
			colOffset: this.colOffset,
			colSize: this.colSize,
		};
		worker.postMessage(message, []);

		

		
	}

}

export class Potree3Loader{

	constructor(){

	}

	static async load(url){

		let rMetadata = await fetch(`${url}/${metadataFilename}`);
		let jsMetadata = await rMetadata.json();

		let batches = [];
		let nodes = [];
		let nodeMap = new Map();

		for(let jsBatch of jsMetadata.batches){
			
			let batch = new Batch();
			batch.name = jsBatch.name;
			batch.url = `${url}/chunks.bin`;
			batch.posOffset = jsBatch.pos_offset;
			batch.posSize = jsBatch.pos_size;
			batch.colOffset = jsBatch.col_offset;
			batch.colSize = jsBatch.col_size;

			let numVoxelsProcessed = 0;

			for(let jsNode of jsBatch.nodes){

				let node = new Node();
				node.name = jsNode.name;
				node.level = node.name.length - 1;
				node.batch = batch;
				node.numVoxels = jsNode.voxels;
				node.numVoxelsProcessed = numVoxelsProcessed;
				numVoxelsProcessed += node.numVoxels;

				batch.nodes.push(node);
				nodes.push(node);
				nodeMap.set(node.name, node);
			}

			batches.push(batch);
		}

		let root = null;

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

		console.log(root);

		root.batch.load();


		let visibleNodes = [];

		potree.onUpdate( () => {

			if(!guiContent["update"]){
				return;
			}

			let numVisibleVoxels = 0;

			root.traverse( (node) => { 
				node.visible = false;
			} );

			let view = camera.view;
			let proj = camera.proj;
			let fm = new Matrix4().multiply(proj).multiply(view); 
			let frustum = new Frustum();
			frustum.setFromMatrix(fm);

			let priorityQueue = new BinaryHeap((element) => {element.weight});
			priorityQueue.push({node: root, weight: 1});

			visibleNodes = [];
			while (priorityQueue.size() > 0) {

				let element = priorityQueue.pop();
				let node = element.node;

				if(!node.batch.loaded){
					node.batch.load();
					node.visible = false;

					continue;
				}

				node.visible = true;
				visibleNodes.push(node);

				for(let child of node.children){

					if (!child) continue;

					let center = child.boundingBox.center();
					let size = child.boundingBox.size().length();
					let camWorldPos = camera.getWorldPosition();
					let distance = camWorldPos.distanceTo(center);
					let weight = (size / distance);
					let visible = weight > 0.3 / Potree.settings.debugU;

					visible = visible && frustum.intersectsBox(child.boundingBox);

					if(visible){
						priorityQueue.push({node: child, weight: weight});
					}
				}
			}

			Potree.state.numVoxels = numVisibleVoxels;
			guiContent["#nodes"] = visibleNodes.length.toLocaleString();

			
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