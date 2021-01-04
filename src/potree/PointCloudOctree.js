
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

				if(loadQueue.length > 8){
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

		for(let node of loadQueue){
			this.load(node);
		}


		this.visibleNodes = visibleNodes;
	}

}