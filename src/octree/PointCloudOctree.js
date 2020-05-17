
import {SceneNode} from "../scene/SceneNode.js";
import {Frustum} from "../math/Frustum.js";
import {Matrix4} from "../math/Matrix4.js";

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

		let view = camera.getView();
		let camWorld = new Matrix4().getInverse(view);
		let proj = camera.getProjection();

		let frustum = new Frustum().setFromProjectionMatrix(proj);
		frustum.applyMatrix4(camWorld);

		let stack = [this.root];
		while(stack.length > 0){
			let node = stack.pop();

			let nodeCenter = node.boundingBox.center();
			let camdist = campos.distanceTo(nodeCenter);
			let nodesize = node.boundingBox.size().length();

			let priority = (Math.tan(camera.fov) * nodesize / 2) / camdist;
			let intersects = frustum.intersectsBox(node.boundingBox);

			let priorityThreshold = window?.debug?.minNodeSize ?? 0.2;
			let visible = priority > priorityThreshold && intersects;


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

			if(i >= 20){
				break;
			}
		}


		return visibleNodes;
	}

	update(state){

		if(!window.debug?.freeze){
			let visibleNodes = this.getVisibleNodes(state.camera);
			this.visibleNodes = visibleNodes;
		}


		if(window.debug?.displayBoxes){
			for(let node of this.visibleNodes){
				state.drawBoundingBox({
					position: node.boundingBox.center(),
					scale: node.boundingBox.size(),
				});
			}
		}

		if(window.debug){
			let visiblePoints = this.visibleNodes.reduce( (a, v) => a + v.numPoints, 0).toLocaleString();
			// visiblePoints = visiblePoints.replace(/\./g, " ");

			window.debug["#nodes"] = this.visibleNodes.length;
			window.debug["#points"] = visiblePoints;

		}



	}

}