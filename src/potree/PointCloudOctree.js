
import {SceneNode} from "../scene/SceneNode.js";
import {PointCloudOctreeNode} from "./PointCloudOctreeNode.js";

export class PointCloudOctree extends SceneNode{

	constructor(name){
		super(name);

		this.loader = null;
		this.root = null;
		this.spacing = 1;
		this.loaded = false;
		this.loading = false;
		this.visibleNodes = [];
		
	}

	load(node){

		if(!node.loaded){
			this.loader.loadNode(node);
		}

	}

	updateVisibility(camera){

		let visibleNodes = [];

		// traverse breadth first
		let loadQueue = [];
		let queue = [this.root];
		while(queue.length > 0){
			let node = queue.shift();

			if(!node.loaded){
				loadQueue.push(node);

				if(loadQueue.length > 10){
					break;
				}

				continue;
			}

			visibleNodes.push(node);

			if(node.level < 4){
				for(let child of node.children){
					if(child){
						queue.push(child);
					}
				}
			}

		}

		for(let node of loadQueue){
			this.load(node);
		}


		this.visibleNodes = visibleNodes;
	}

	updateVisibility1(camera){

		let visibleNodes = [];

		// traverse breadth first
		let loadQueue = [];
		let queue = [this.root];
		while(queue.length > 0){
			let node = queue.shift();

			if(!node.loaded){
				loadQueue.push(node);

				if(loadQueue.length > 10){
					break;
				}

				continue;
			}

			visibleNodes.push(node);

			if(node.level < 4){
				for(let child of node.children){
					if(child){
						queue.push(child);
					}
				}
			}

		}

		for(let node of loadQueue){
			this.load(node);
		}


		this.visibleNodes = visibleNodes;

		// if(loadQueue.length >= 4){
			
		// 	loadQueue.sort((a, b) => {

		// 		if(a.byteOffset == null || b.byteOffset == null){
		// 			return -1;
		// 		}

		// 		return Number(a.byteOffset - b.byteOffset);
		// 	});

		// 	let first = {
		// 		byteOffset: loadQueue[0].byteOffset,
		// 		byteSize: loadQueue[0].byteSize,
		// 		nodes: [loadQueue[0]],
		// 	};
		// 	let batches = [first];
		// 	for(let i = 1; i < loadQueue.length; i++){
				
		// 		let a = batches[batches.length - 1];
		// 		let b = first = {
		// 			byteOffset: loadQueue[i].byteOffset,
		// 			byteSize: loadQueue[i].byteSize,
		// 			nodes: [loadQueue[i]],
		// 		};

		// 		if(a.byteOffset == null || b.byteOffset == null){
		// 			continue;
		// 		}
				
		// 		// if(a.byteOffset + a.byteSize === b.byteOffset){
		// 		if(Math.abs(Number((a.byteOffset + a.byteSize) - b.byteOffset)) < 100_000 ){
		// 			// merge
		// 			a.byteSize += b.byteSize;
		// 			a.nodes.push(b);
		// 		}else{
		// 			batches.push(b);
		// 		}
				
				
		// 	}
		// 	console.log(`${loadQueue.length} => ${batches.length}`);
		// }

		
	}

}