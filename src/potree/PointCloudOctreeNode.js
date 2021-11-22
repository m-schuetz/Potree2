
import {Box3} from "../math/Box3.js";

export class PointCloudOctreeNode{
	constructor(name){
		this.name = name;
		this.loaded = false;
		this.parent = null;
		this.children = new Array(8).fill(null);
		this.level = 0;
		this.numPoints = 0;

		this.boundingBox = new Box3();
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