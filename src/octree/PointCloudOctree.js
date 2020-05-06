
import {SceneNode} from "../scene/SceneNode.js";

export class Node{

	constructor(){
		this.buffers = null;
		this.boundingBox = null;
		this.children = [
			null, null, null, null,
			null, null, null, null,
		];
	}

}


export class PointCloudOctree extends SceneNode{

	constructor(){
		super("");

		this.loader = null;
		this.root = null;
		this.boundingBox = null;
		this.visibleNodes = [];
		
	}

	update(state){

		state.drawBoundingBox({
			position: this.boundingBox.center(),
			scale: this.boundingBox.size(),
		});

		let visibleNodes = [];
		let nodesToLoad = [];

		let stack = [this.root];
		while(stack.length > 0){
			let node = stack.pop();

			let visible = node.loaded;

			if(!node.loaded){
				nodesToLoad.push(node);
			}

			if(visible){
				visibleNodes.push(node);

				for(let child of node.children){
					if(child){
						stack.push(child);
					}
				}
			}
		}

		// for(let i = 0; i < nodesToLoad.length; i++){
		// 	let node = nodesToLoad[i];
		// 	this.loader.loadNode(node);

		// 	if(i >= 3){
		// 		break;
		// 	}
		// }

		for(let node of visibleNodes){
			state.drawBoundingBox({
				position: node.boundingBox.center(),
				scale: node.boundingBox.size(),
			});
		}

		this.visibleNodes = visibleNodes;



	}

}