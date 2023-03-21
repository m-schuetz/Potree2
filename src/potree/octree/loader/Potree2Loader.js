
import {PointCloudOctree, PointCloudOctreeNode} from "potree";
import {PointAttribute, PointAttributes, PointAttributeTypes} from "potree";
import {WorkerPool} from "potree";
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

let MAX_BATCHES_LOADING = 10;
let MAX_LEAVES_LOADING  = 10;

export class Potree2Loader{

	static numBatchesLoading = 0;
	static numLeavesLoading = 0;

	constructor(){
		this.metadata = null;
		this.metanodeMap = new Map();
		this.batchnodeMap = new Map();
		this.nodeMap = new Map();
		this.batches = new Map();
		this.octree = null;

		for(let i = 0; i < 10; i++){
			let workerPath = "./src/potree/octree/loader/DecoderWorker_Potree2Batch.js";
			let worker = WorkerPool.getWorker(workerPath, {type: "module"});
			WorkerPool.returnWorker(workerPath, worker);
		}
	}

	async loadPoints(node){

		if(Potree2Loader.numLeavesLoading >= MAX_LEAVES_LOADING){
			return;
		}

		if(node.loading) return;

		node.loading = true;
		Potree2Loader.numLeavesLoading++;

		let metanode = this.metanodeMap.get(node.name);

		let workerPath = "./src/potree/octree/loader/DecoderWorker_points.js";
		let worker = WorkerPool.getWorker(workerPath, {type: "module"});

		worker.onmessage = (e) => {

			let geometry = new Geometry();

			geometry.numElements = metanode.numPoints;
			geometry.buffer = new Uint8Array(e.data.buffer);
			node.geometry = geometry;

			node.loaded = true;
			node.loading = false;

			WorkerPool.returnWorker(workerPath, worker);
			Potree2Loader.numLeavesLoading--;
		};

		let message = {
			url:       new URL(`${node.octree.url}/${node.name}.points`, self.location).href,
			numPoints: metanode.numPoints,
		};

		worker.postMessage(message, []);

	}

	async loadBatch(node){

		if(Potree2Loader.numBatchesLoading >= MAX_BATCHES_LOADING){
			return;
		}

		let batch = node.batch;

		if(batch.loading === true) return;
		if(batch.loaded === true) return;
		
		Potree2Loader.numBatchesLoading++;
		batch.loading = true;

		batch.nodes.sort((a, b) => {
			return a.name.length - b.name.length;
		});

		{
			let workerPath = "./src/potree/octree/loader/DecoderWorker_Potree2Batch.js";
			let worker = WorkerPool.getWorker(workerPath, {type: "module"});

			worker.onmessage = (e) => {

				let data = e.data;


				if(data.type === "node parsed"){
					let workernode = data.node;
					
					let geometry = new Geometry();
					geometry.numElements = workernode.numVoxels;
					geometry.buffer = new Uint8Array(data.buffer);

					let node = this.nodeMap.get(workernode.name);
					node.loaded = true;
					node.loading = false;
					node.geometry = geometry;
				}else if(data.type === "batch finished"){
					WorkerPool.returnWorker(workerPath, worker);

					batch.loaded = true;
					batch.loading = false;
					Potree2Loader.numBatchesLoading--;
				}
			};

			let url = new URL(`${this.octree.url}/${batch.name}.batch`, document.baseURI).href;

			let parent = this.nodeMap.get(batch.name).parent;

			let parentMsg = null;
			if(parent){
				let parentMetanode = this.metanodeMap.get(parent.name);
				let parentBMetanode = this.batchnodeMap.get(parent.name);
				parentMsg = {
					name: parent.name,
					min: parent.boundingBox.min.toArray(),
					max: parent.boundingBox.max.toArray(),
					numVoxels: parentMetanode.numVoxels,
					buffer: parent.geometry.buffer.buffer.slice(),
					numChildVoxels: parentBMetanode.numChildVoxels,
					numVoxelsPerOctant: parentBMetanode.numVoxelsPerOctant,
				};
			}

			// debugger;

			let nodes = batch.nodes;
			let spacing = this.octree.spacing;

			let message = {
				batchName: batch.name,
				url, nodes, spacing,
				parent: parentMsg,
			};

			// debugger;

			worker.postMessage(message, []);

		}

		// batch.loading = false;
		// batch.loaded = true;
	}

	async loadNode(node){

		let metanode = this.metanodeMap.get(node.name);

		if(metanode.numVoxels > 0){
			this.loadBatch(node);
		}else if(metanode.numPoints > 0){
			this.loadPoints(node);
		}

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

		// LOAD NODES
		for(let metanode of metadata.nodes){

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

			loader.nodeMap.set(node.name, node);
			loader.metanodeMap.set(node.name, metanode);
		}

		// connect nodes
		for(let [nodename, node] of loader.nodeMap){

			if(nodename === "r") continue;

			let parentName = nodename.substr(0, nodename.length - 1);
			let parent = loader.nodeMap.get(parentName);

			let childIndex = Number(nodename.slice(-1));
			parent.children[childIndex] = node;
			node.parent = parent;
		}

		// BATCHES
		for(let batch of metadata.batches){
			loader.batches.set(batch.name, batch);

			for(let bmetanode of batch.nodes){
				let node = loader.nodeMap.get(bmetanode.name);

				node.batch = batch;

				loader.batchnodeMap.set(node.name, bmetanode);
			}
		}

		let root = loader.nodeMap.get("r");

		let nodes = Array.from(loader.nodeMap.values());
		nodes.sort((a, b) => {
			return a.level - b.level;
		});

		octree.root = root;

		Potree.events.dispatcher.dispatch("pointcloud_loaded", octree);

		return octree;
	}

}