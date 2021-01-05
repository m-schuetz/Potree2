
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

		// if(this.root.geometry){
		// 	visibleNodes.push(this.root);
		// }

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

			if(node.level < 5){
				for(let child of node.children){
					if(child){
						queue.push(child);
					}
				}
			}

		}

		if(loadQueue.length >= 16){
			
			loadQueue.sort((a, b) => {

				if(a.byteOffset == null || b.byteOffset == null){
					return -1;
				}

				return Number(a.byteOffset - b.byteOffset);
			});

			let first = {
				byteOffset: loadQueue[0].byteOffset,
				byteSize: loadQueue[0].byteSize,
				nodes: [loadQueue[0]],
			};
			let batches = [first];
			for(let i = 1; i < loadQueue.length; i++){
				
				let a = batches[batches.length - 1];
				let b = first = {
					byteOffset: loadQueue[i].byteOffset,
					byteSize: loadQueue[i].byteSize,
					nodes: [loadQueue[i]],
				};
				
				if(a.byteOffset + a.byteSize === b.byteOffset){
					// merge
					a.byteSize += b.byteSize;
					a.nodes.push(b);
				}else{
					batches.push(b);
				}
				
				
			}
			console.log(`${loadQueue.length} => ${batches.length}`);
		}

		for(let node of loadQueue){
			this.load(node);
		}


		this.visibleNodes = visibleNodes;
	}

}