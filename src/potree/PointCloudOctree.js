
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
		
	}

	load(node){

		if(!node.loaded){
			this.loader.loadNode(node);
		}

	}

	updateVisibility(camera){

	}

}