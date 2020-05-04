
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

}