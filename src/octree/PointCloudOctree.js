
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

	getVisibleNodes(camera){
		let visibleNodes = [];
		let nodesToLoad = [];

		let campos = camera.position;

		let stack = [this.root];
		while(stack.length > 0){
			let node = stack.pop();

			let nodeCenter = node.boundingBox.center();
			let camdist = campos.distanceTo(nodeCenter);
			let nodesize = node.boundingBox.size().length();

			let priority = (Math.tan(camera.fov) * nodesize / 2) / camdist;
			
			let visible = priority > 0.1;

			if(visible && !node.loaded){
				nodesToLoad.push({
					node: node,
					priority: priority,
				});
			}

			if(visible && node.loaded){
				visibleNodes.push(node);

				for(let child of node.children){
					if(child){
						stack.push(child);
					}
				}
			}
		}

		nodesToLoad.sort( (a, b) => b.priority - a.priority);

		for(let i = 0; i < nodesToLoad.length; i++){
			let item = nodesToLoad[i];
			this.loader.loadNode(item.node);

			if(i >= 3){
				break;
			}
		}


		return visibleNodes;
	}

	update(state){

		let visibleNodes = this.getVisibleNodes(state.camera);

		// for(let node of visibleNodes){
		// 	state.drawBoundingBox({
		// 		position: node.boundingBox.center(),
		// 		scale: node.boundingBox.size(),
		// 	});
		// }

		this.visibleNodes = visibleNodes;



	}

}